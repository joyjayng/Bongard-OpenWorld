import logging

import hydra
import pytorch_lightning as pl
from lightning import seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import datasets
import models
import wandb

logger = logging.getLogger(__name__)


class BongardCallback(Callback):
    def __init__(self, use_clip_stat=False):
        super().__init__()
        if use_clip_stat:
            # CLIP
            self.pix_mean = (0.48145466, 0.4578275, 0.40821073)
            self.pix_std = (0.26862954, 0.26130258, 0.27577711)
        else:
            # IN
            self.pix_mean = (0.485, 0.456, 0.406)
            self.pix_std = (0.229, 0.224, 0.225)

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log('acc/train_bongard_acc',
                      pl_module.train_acc.avg, sync_dist=True)
        pl_module.log('acc/train_bongard_acc_concept_2',
                      pl_module.train_acc_concept_2.avg, sync_dist=True)
        pl_module.log('acc/train_bongard_acc_concept_3',
                      pl_module.train_acc_concept_3.avg, sync_dist=True)
        pl_module.log('acc/train_bongard_acc_concept_4',
                      pl_module.train_acc_concept_4.avg, sync_dist=True)
        pl_module.log('acc/train_bongard_acc_concept_5',
                      pl_module.train_acc_concept_5.avg, sync_dist=True)
        pl_module.log('acc/train_bongard_acc_commonsense',
                      pl_module.train_acc_commonsense.avg, sync_dist=True)
        pl_module.log('acc/train_bongard_acc_non_commonsense',
                      pl_module.train_acc_non_commonsense.avg, sync_dist=True)
        pl_module.log(
            'misc/train_lr', pl_module.optimizers()[0].param_groups[0]['lr'], sync_dist=True)
        pl_module.log(
            'misc/train_lr_image_encoder', pl_module.optimizers()[1].param_groups[0]['lr'], sync_dist=True)
        pl_module.train_acc.reset()
        pl_module.train_acc_concept_2.reset()
        pl_module.train_acc_concept_3.reset()
        pl_module.train_acc_concept_4.reset()
        pl_module.train_acc_concept_5.reset()
        pl_module.train_acc_commonsense.reset()
        pl_module.train_acc_non_commonsense.reset()

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.log('acc/val_bongard_acc',
                      pl_module.val_acc.avg, sync_dist=True)
        pl_module.log('acc/val_bongard_acc_concept_2',
                      pl_module.val_acc_concept_2.avg, sync_dist=True)
        pl_module.log('acc/val_bongard_acc_concept_3',
                      pl_module.val_acc_concept_3.avg, sync_dist=True)
        pl_module.log('acc/val_bongard_acc_concept_4',
                      pl_module.val_acc_concept_4.avg, sync_dist=True)
        pl_module.log('acc/val_bongard_acc_concept_5',
                      pl_module.val_acc_concept_5.avg, sync_dist=True)
        pl_module.log('acc/val_bongard_acc_commonsense',
                      pl_module.val_acc_commonsense.avg, sync_dist=True)
        pl_module.log('acc/val_bongard_acc_non_commonsense',
                      pl_module.val_acc_non_commonsense.avg, sync_dist=True)

        logs = []
        for img, cap in pl_module.viz_caption_buffer:
            logs.append((img * self.pix_std + self.pix_mean, cap))

        pl_module.logger.log_image(key='example/val_caption',
                                   images=[i[0][..., ::-1] for i in logs],
                                   caption=[i[1] for i in logs])
        pl_module.val_acc.reset()
        pl_module.val_acc_concept_2.reset()
        pl_module.val_acc_concept_3.reset()
        pl_module.val_acc_concept_4.reset()
        pl_module.val_acc_concept_5.reset()
        pl_module.val_acc_commonsense.reset()
        pl_module.val_acc_non_commonsense.reset()
        pl_module.viz_caption_buffer = []

    def on_test_epoch_end(self, trainer, pl_module):
        logging.info('Test end!')
        pl_module.log('acc/test_bongard_acc',
                      pl_module.test_acc.avg, sync_dist=True)
        pl_module.log('acc/test_bongard_acc_concept_2',
                      pl_module.test_acc_concept_2.avg, sync_dist=True)
        pl_module.log('acc/test_bongard_acc_concept_3',
                      pl_module.test_acc_concept_3.avg, sync_dist=True)
        pl_module.log('acc/test_bongard_acc_concept_4',
                      pl_module.test_acc_concept_4.avg, sync_dist=True)
        pl_module.log('acc/test_bongard_acc_concept_5',
                      pl_module.test_acc_concept_5.avg, sync_dist=True)
        pl_module.log('acc/test_bongard_acc_commonsense',
                      pl_module.test_acc_commonsense.avg, sync_dist=True)
        pl_module.log('acc/test_bongard_acc_non_commonsense',
                      pl_module.test_acc_non_commonsense.avg, sync_dist=True)
        logger.info('#wrong answer: %d', len(pl_module.wrong_q))
        logging.info(f"""
            All metrics:
             test_bongard_acc: {pl_module.test_acc.avg}
             test_bongard_acc_concept_2: {pl_module.test_acc_concept_2.avg}
             test_bongard_acc_concept_3: {pl_module.test_acc_concept_3.avg}
             test_bongard_acc_concept_4: {pl_module.test_acc_concept_4.avg}
             test_bongard_acc_concept_5: {pl_module.test_acc_concept_5.avg}
             test_bongard_acc_concept_commonsense: {pl_module.test_acc_commonsense.avg}
             test_bongard_acc_concept_non_commonsense: {pl_module.test_acc_non_commonsense.avg}
             test_bongard_acc_pos: {pl_module.test_acc_pos.avg}
             test_bongard_acc_neg: {pl_module.test_acc_neg.avg}
        """)
        # f = open('wrong_q_test.txt', 'w')
        # for uid in pl_module.wrong_q:
        #     f.write(str(uid)+'\n')
        # f.close()


@hydra.main(version_base=None, config_path='configs', config_name='bongard')
def main(cfg):
    logging.basicConfig(level=logging.INFO)
    logger.info('Config: %s', cfg)
    seed_everything(cfg.seed, workers=True)
    kwargs = {}
    kwargs['logger'] = WandbLogger(project='bongard-ow', name=cfg.exp_name)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor='acc/val_bongard_acc',
        mode='max',
        filename='bongard-ow-{epoch:02d}-{acc/val_bongard_acc:.2f}',
    )

    callbacks = [
        BongardCallback(use_clip_stat=cfg.val_dataset_args.use_clip_stat),
        checkpoint_callback,
    ]
    trainer = pl.Trainer(
        max_epochs=cfg.max_epoch,
        devices=cfg.devices,
        precision=cfg.precision,
        accelerator=cfg.accelerator,
        strategy=cfg.strategy,
        callbacks=callbacks,
        log_every_n_steps=10,
        **kwargs,
    )

    # model
    model = models.bongard_model.BongardModel(
        cfg.image_encoder,
        cfg.image_encoder_output_dim,
        cfg.image_encoder_args,
        cfg.relational_encoder,
        cfg.few_shot_head,
        cfg.few_shot_head_args,
        cfg.caption_head_name,
        cfg.caption_head_args,
        cfg.optimizer_name,
        cfg.optimizer_args,
        cfg.bongard_args,
        cfg.freeze_image_encoder,
        cfg.caption_loss_coeff,
        cfg.additional_args
    )

    # dataset
    train_dataset = datasets.make(cfg.train_dataset, **cfg.train_dataset_args)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dataset_workers,
        shuffle=True,
        collate_fn=datasets.bongard_ow.collate_images_boxes_dict,
        pin_memory=True,
    )
    val_dataset = datasets.make(cfg.val_dataset, **cfg.val_dataset_args)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dataset_workers,
        shuffle=False,
        collate_fn=datasets.bongard_ow.collate_images_boxes_dict,
        pin_memory=True,
    )
    test_dataset = datasets.make(cfg.test_dataset, **cfg.test_dataset_args)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dataset_workers,
        shuffle=False,
        collate_fn=datasets.bongard_ow.collate_images_boxes_dict,
        pin_memory=True,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.ckpt_path
    )

    # trainer.test(model, dataloaders=test_loader, ckpt_path=cfg.ckpt_path)
    logger.info('Test start!')
    logger.info('Testing with the best model ' + checkpoint_callback.best_model_path)
    trainer.test(model, dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path)

if __name__ == '__main__':
    main()
