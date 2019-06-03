import torch
from torch import nn
from dataset import GenerateIterator, GenerateIterator_eval
from myargs import args
import numpy as np
from tqdm import tqdm
from models import Classifier, Discriminator, EMA
from vat import VAT, ConditionalEntropyLoss


# discriminator network
feature_discriminator = Discriminator(large=args.large).cuda()

# classifier network.
classifier = Classifier(large=args.large).cuda()

# loss functions
cent = ConditionalEntropyLoss().cuda()
xent = nn.CrossEntropyLoss(reduction='mean').cuda()
sigmoid_xent = nn.BCEWithLogitsLoss(reduction='mean').cuda()
vat_loss = VAT(classifier).cuda()

# optimizer.
optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_disc = torch.optim.Adam(feature_discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# datasets.
iterator_train = GenerateIterator(args)
iterator_val = GenerateIterator_eval(args)

# loss params.
dw = 1e-2
cw = 1
sw = 1
tw = 1e-2
bw = 1e-2


''' Exponential moving average (simulating teacher model) '''
ema = EMA(0.998)
ema.register(classifier)

# training..
for epoch in range(1, args.num_epoch):
    iterator_train.dataset.shuffledata()
    pbar = tqdm(iterator_train, disable=False,
                bar_format="{percentage:.0f}%,{elapsed},{remaining},{desc}")

    loss_main_sum, n_total = 0, 0
    loss_domain_sum, loss_src_class_sum, \
    loss_src_vat_sum, loss_trg_cent_sum, loss_trg_vat_sum = 0, 0, 0, 0, 0
    loss_disc_sum = 0

    for images_source, labels_source, images_target, labels_target in pbar:
        images_source, labels_source, images_target, labels_target = images_source.cuda(), labels_source.cuda(), images_target.cuda(), labels_target.cuda()

        # pass images through the classifier network.
        feats_source, pred_source = classifier(images_source)
        feats_target, pred_target = classifier(images_target, track_bn=True)

        ' Discriminator losses setup. '
        # discriminator loss.
        real_logit_disc = feature_discriminator(feats_source.detach())
        fake_logit_disc = feature_discriminator(feats_target.detach())

        loss_disc = 0.5 * (
                sigmoid_xent(real_logit_disc, torch.ones_like(real_logit_disc, device='cuda')) +
                sigmoid_xent(fake_logit_disc, torch.zeros_like(fake_logit_disc, device='cuda'))
        )

        ' Classifier losses setup. '
        # supervised/source classification.
        loss_src_class = xent(pred_source, labels_source)

        # conditional entropy loss.
        loss_trg_cent = cent(pred_target)

        # virtual adversarial loss.
        loss_src_vat = vat_loss(images_source, pred_source)
        loss_trg_vat = vat_loss(images_target, pred_target)

        # domain loss.
        real_logit = feature_discriminator(feats_source)
        fake_logit = feature_discriminator(feats_target)

        loss_domain = 0.5 * (
                sigmoid_xent(real_logit, torch.zeros_like(real_logit, device='cuda')) +
                sigmoid_xent(fake_logit, torch.ones_like(fake_logit, device='cuda'))
        )

        # combined loss.
        loss_main = (
                dw * loss_domain +
                cw * loss_src_class +
                sw * loss_src_vat +
                tw * loss_trg_cent +
                tw * loss_trg_vat
        )

        ' Update network(s) '

        # Update discriminator.
        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

        # Update classifier.
        optimizer_cls.zero_grad()
        loss_main.backward()
        optimizer_cls.step()

        # Polyak averaging.
        ema(classifier)  # TODO: move ema into the optimizer step fn.

        loss_domain_sum += loss_domain.item()
        loss_src_class_sum += loss_src_class.item()
        loss_src_vat_sum += loss_src_vat.item()
        loss_trg_cent_sum += loss_trg_cent.item()
        loss_trg_vat_sum += loss_trg_vat.item()
        loss_main_sum += loss_main.item()
        loss_disc_sum += loss_disc.item()
        n_total += 1

        pbar.set_description('loss {:.3f},'
                             ' domain {:.3f},'
                             ' s cls {:.3f},'
                             ' s vat {:.3f},'
                             ' t c-ent {:.3f},'
                             ' t vat {:.3f},'
                             ' disc {:.3f}'.format(
            loss_main_sum/n_total,
            loss_domain_sum/n_total,
            loss_src_class_sum/n_total,
            loss_src_vat_sum/n_total,
            loss_trg_cent_sum/n_total,
            loss_trg_vat_sum/n_total,
            loss_disc_sum / n_total,
        )
    )

    # validate.
    if epoch % 1 == 0:
        classifier.eval()
        feature_discriminator.eval()

        with torch.no_grad():
            preds_val, gts_val = [], []
            val_loss = 0
            for images_target, labels_target in iterator_val:
                images_target, labels_target = images_target.cuda(), labels_target.cuda()

                # cross entropy based classification
                _, pred_val = classifier(images_target)

                pred_val = np.argmax(pred_val.cpu().data.numpy(), 1)

                preds_val.extend(pred_val)
                gts_val.extend(labels_target)

            preds_val = np.asarray(preds_val)
            gts_val = np.asarray(gts_val)

            score_cls_val = (np.mean(preds_val == gts_val)).astype(np.float)
            print('\n({}) acc. v {:.3f}\n'.format(epoch, score_cls_val))

        feature_discriminator.train()
        classifier.train()
