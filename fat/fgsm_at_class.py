import copy
import os
import time
import random
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torchattacks import PGD, FGSM
import torch.nn.functional as F
from .at_base import ATBase


def one_hot_encode(labels, num_classes):
    return torch.zeros(labels.size(0), num_classes, device=labels.device).scatter_(
        1, labels.unsqueeze(1), 1
    )


def self_guided_label_smooth_encode(logits, labels, smoothing, temperature=8.0):
    num_classes = logits.size(1)
    with torch.no_grad():
        soft_probs = torch.softmax(logits / temperature, dim=1)
    one_hot = one_hot_encode(labels, num_classes)
    soft_targets = (1.0 - smoothing) * one_hot + smoothing * soft_probs
    return soft_targets


def soft_cross_entropy(logits, soft_targets):
    soft_targets = soft_targets.float()
    log_softmax = F.log_softmax(logits, dim=1)
    loss = torch.sum(-soft_targets * log_softmax, dim=1).mean()
    return loss


class FGSM_AT_Class(ATBase):

    def __init__(
        self,
        model,
        eps=8.0 / 255,
        log_dir="./log_cifar10/",
        name="at-fat-class",
        device=None,
        seed=0,
        num_class=10,
    ):
        super(FGSM_AT_Class, self).__init__(
            model, eps=eps, log_dir=log_dir, name=name, device=device, seed=seed
        )
        self.momentum_decay = 0.3
        self.lamda = 10
        self.uap_eps = 10.0 / 255
        self.num_class = num_class

    def generate_label_info(self, num_classes, image_shape, device, eps, criterion):
        self.model.eval()
        x_gray = torch.ones((num_classes, *image_shape), device=device) * 0.5
        x_gray.requires_grad = True
        labels = torch.arange(num_classes, device=device)

        outputs = self.model(x_gray)
        loss = criterion(outputs, labels)
        loss.backward()
        grad = x_gray.grad.detach()
        L_y = eps * torch.sign(grad)
        return L_y

    def train(
        self,
        opt,
        scheduler,
        train_loader,
        test_loader,
        total_epoch=110,
        label_smoothing=0.4,
        weight_average=True,
        tau=0.9995,
        image_shape=(3, 32, 32),
        eval_start=90,
        alpha=0.6,
        T=8.0,
    ):
        if label_smoothing is not None:
            criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            criterion = CrossEntropyLoss()

        loss_fn = MSELoss()
        num_classes = self.num_class
        uaps = torch.zeros((num_classes, *image_shape)).uniform_(-self.uap_eps, self.uap_eps).to(self.device)
        uaps = torch.clamp(self.uap_eps * torch.sign(uaps), -self.uap_eps, self.uap_eps)

        momentum = torch.ones((num_classes, *image_shape)).to(self.device) * 0.9
        L_y_all = torch.zeros((num_classes, *image_shape), device=self.device)

        if weight_average:
            wa_model = copy.deepcopy(self.model)
            exp_avg = self.model.state_dict()
            if tau is None:
                raise ValueError("tau should not be None when weight_average is True")
            pgd_attacker = PGD(wa_model, eps=self.eps, alpha=2.0 / 255, steps=10)
            fgsm_attacker = FGSM(wa_model, eps=self.eps)
        else:
            pgd_attacker = PGD(self.model, eps=self.eps, alpha=2.0 / 255, steps=10)
            fgsm_attacker = FGSM(self.model, eps=self.eps)

        self.logger.log("scheduler: {}".format(scheduler.__class__.__name__))
        self.logger.log("max smoothing: {}".format(label_smoothing))
        self.logger.log("weight average: {}, tau: {}".format(weight_average, tau))
        self.logger.new_line()
        self.logger.new_line()

        best_pgd_acc, best_test_acc, total_training_time, total_test_time = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        train_loss_list = []
        train_acc_list, test_acc_list, pgd_acc_list, fgsm_acc_list = [], [], [], []
        cur_lambda, cur_update = 0, self.lamda / total_epoch

        for epoch in range(total_epoch):
            self.logger.log("============ Epoch {} ============".format(epoch))
            self.model.train()
            train_loss, train_correct, train_n = 0, 0, 0
            start_time = time.time()
            L_y_now = self.generate_label_info(
                num_classes=num_classes,
                image_shape=image_shape,
                device=self.device,
                eps=5.0 / 255,
                criterion=criterion,
            )
            L_y_all = 0.75 * L_y_all + (1 - 0.75) * L_y_now
            max_smoothing = label_smoothing
            cur_smoothing = max_smoothing * epoch / total_epoch
            cur_lambda += cur_update
            
            self.logger.log("cur_smoothing: {}".format(cur_smoothing))
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                B, C_img, H, W = images.shape
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(images)
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[torch.arange(B), labels] = False
                wrong_logits = logits.masked_fill(~mask, float("-inf"))
                target_labels = torch.argmax(wrong_logits, dim=1)
                self.model.train()  # 切换回 train 模式

                L_y_batch = L_y_all[labels]  # shape: [batch_size, C, H, W]
                L_t_batch = L_y_all[target_labels]
                init = (
                    uaps[labels].clone()
                    - L_y_batch
                    + L_t_batch
                )
                init = torch.clamp(init, -self.uap_eps, self.uap_eps)
                adv_images = images + init

                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
                adv_images.requires_grad_(True)

                ori_output, _ = self.model(adv_images, mid_fea=True)
                targeted_loss = alpha * criterion(ori_output, labels) - (
                    1 - alpha
                ) * criterion(ori_output, target_labels)

                targeted_loss.backward(retain_graph=True)
                grad_x = adv_images.grad.detach()
                adv_images = adv_images.detach() + self.eps * grad_x.sign()

                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
                adv_images.requires_grad_(False)

                output, _ = self.model(adv_images, mid_fea=True)

                loss_reg = loss_fn(output.float(), ori_output.float())
                soft_targets = self_guided_label_smooth_encode(
                    output.detach(), labels, cur_smoothing, T
                )

                loss_ce = soft_cross_entropy(output, soft_targets)
                loss = loss_ce + self.lamda * loss_reg

                opt.zero_grad()
                loss.backward()
                if epoch < 1:
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                opt.step()

                grad_norm = torch.norm(grad_x, p=1)
                cur_grad = grad_x / grad_norm
                for class_idx in set(labels.tolist()):
                    momentum[class_idx] = cur_grad[class_idx == labels].mean(dim=0) + momentum[class_idx] * 0.3
                    uaps[class_idx] = torch.clamp(
                        uaps[class_idx] + self.uap_eps * torch.sign(momentum[class_idx]), -self.uap_eps, self.uap_eps)

                momentum = momentum.detach()
                uaps = uaps.detach()

                # weight average
                if weight_average:
                    for key, value in self.model.state_dict().items():
                        exp_avg[key] = (1 - tau) * value + tau * exp_avg[key]

                train_loss += loss.item() * labels.size(0)
                train_correct += (output.max(1)[1] == labels).sum().item()
                train_n += labels.size(0)

            scheduler.step()
            epoch_train_loss = train_loss / train_n
            train_loss_list.append(epoch_train_loss)

            if weight_average:
                self.model.eval()
                wa_model.load_state_dict(exp_avg)
                wa_model.eval()
            else:
                self.model.eval()

            total_training_time += time.time() - start_time
            self.logger.log("Training time: {:.2f}".format(time.time() - start_time))
            self.logger.log("Training loss: {:.4f}".format(train_loss / train_n))
            self.logger.log("Training accuracy: {:.4f}".format(train_correct / train_n))
            train_acc_list.append(train_correct / train_n)

            if epoch < eval_start:
                self.logger.new_line()
                continue

            start_time = time.time()
            test_correct, fgsm_correct, pgd_correct, test_num = 0, 0, 0, 0

            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # clean accuracy
                if weight_average:
                    output = wa_model(images)
                else:
                    output = self.model(images)
                test_correct += (output.max(1)[1] == labels).sum().item()
                test_num += labels.size(0)

                # pgd accuracy
                pgd_attacker.model = wa_model if weight_average else self.model
                adv_images = pgd_attacker(images, labels)
                if weight_average:
                    output = wa_model(adv_images)
                else:
                    output = self.model(adv_images)
                pgd_correct += (output.max(1)[1] == labels).sum().item()

                # fgsm accuracy
                fgsm_attacker.model = wa_model if weight_average else self.model
                fgsm_images = fgsm_attacker(images, labels)
                if weight_average:
                    output = wa_model(fgsm_images)
                else:
                    output = self.model(fgsm_images)
                fgsm_correct += (output.max(1)[1] == labels).sum().item()

            total_test_time += time.time() - start_time
            self.logger.log("Test time: {:.2f}".format(time.time() - start_time))
            self.logger.log("Test accuracy: {:.4f}".format(test_correct / test_num))
            self.logger.log("FGSM accuracy: {:.4f}".format(fgsm_correct / test_num))
            self.logger.log("PGD accuracy: {:.4f}".format(pgd_correct / test_num))

            test_acc_list.append(test_correct / test_num)
            pgd_acc_list.append(pgd_correct / test_num)
            fgsm_acc_list.append(fgsm_correct / test_num)

            if pgd_correct / test_num > best_pgd_acc or (
                pgd_correct / test_num == best_pgd_acc
                and test_correct / test_num > best_test_acc
            ):
                best_pgd_acc = pgd_correct / test_num
                best_test_acc = test_correct / test_num
                if weight_average:
                    torch.save(
                        wa_model.state_dict(), os.path.join(self.output_dir, "best.pth")
                    )
                else:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.output_dir, "best.pth"),
                    )

            self.logger.new_line()
        self.logger.log("train_acc_list: \n" + str(train_acc_list))
        self.logger.new_line()
        self.logger.log("test_acc_list: \n" + str(test_acc_list))
        self.logger.new_line()
        self.logger.log("pgd_acc_list: \n" + str(pgd_acc_list))
        self.logger.new_line()
        self.logger.log("fgsm_acc_list: \n" + str(fgsm_acc_list))
        self.logger.new_line()
        self.logger.log("total_training_time: \n" + str(total_training_time))
        self.logger.new_line()
        self.logger.log("total_test_time: \n" + str(total_test_time))
        self.logger.new_line()

        if weight_average:
            torch.save(wa_model.state_dict(), os.path.join(self.output_dir, "last.pth"))
            torch.save(
                self.model.state_dict(), os.path.join(self.output_dir, "ori_last.pth")
            )
        else:
            torch.save(
                self.model.state_dict(), os.path.join(self.output_dir, "last.pth")
            )
        metrics_data = {
            "train_loss": train_loss_list,
            "train_acc": train_acc_list,
            "test_acc": test_acc_list,
            "pgd_acc": pgd_acc_list,
            "fgsm_acc": fgsm_acc_list,
        }
        self.save_metrics_results(metrics_data, eval_start)
