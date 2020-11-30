import gc

import numpy as np
import torch
from tqdm import tqdm
from text_model_boris.utils.evaluation import target_metric


def train_loop(model, train_loader, optimizer, criterion, scheduler, args, iteration):
    model.train()

    avg_loss = 0.0

    optimizer.zero_grad()
    for idx, batch in enumerate(tqdm(train_loader, desc="Train")):
        x_feats, text1_ids, text2_ids, seg_text1_ids, seg_text2_ids, attn_text1, attn_text2, labels = batch
        x_feats, text1_ids, text2_ids, seg_text1_ids, seg_text2_ids, attn_text1, attn_text2, labels = (
            x_feats.cuda(),
            text1_ids.cuda(),
            text2_ids.cuda(),
            seg_text1_ids.cuda(),
            seg_text2_ids.cuda(),
            attn_text1.cuda(),
            attn_text2.cuda(),
            labels.cuda(),
        )

        logits = model(
            input_ids_text1=text1_ids.long(),
            attention_mask_text1=attn_text1,
            token_type_ids_text1=seg_text1_ids,
            input_ids_text2=text2_ids.long(),
            attention_mask_text2=attn_text2,
            token_type_ids_text2=seg_text2_ids,
        )
        loss = criterion(logits, labels)

        loss.backward()
        if (iteration + 1) % args.batch_accumulation == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        iteration += 1

        avg_loss += loss.item() / (len(train_loader) * args.batch_accumulation)

    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss, iteration


def evaluate(args, model, val_loader, criterion, val_shape):
    avg_val_loss = 0.0
    model.eval()

    valid_preds = np.zeros((val_shape, args.num_classes))
    original = np.zeros((val_shape, args.num_classes))

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Valid")):
            x_feats, text1_ids, text2_ids, seg_text1_ids, seg_text2_ids, attn_text1, attn_text2, labels = batch
            x_feats, text1_ids, text2_ids, seg_text1_ids, seg_text2_ids, attn_text1, attn_text2, labels = (
                x_feats.cuda(),
                text1_ids.cuda(),
                text2_ids.cuda(),
                seg_text1_ids.cuda(),
                seg_text2_ids.cuda(),
                attn_text1.cuda(),
                attn_text2.cuda(),
                labels.cuda(),
            )
            logits = model(
                input_ids_text1=text1_ids.long(),
                attention_mask_text1=attn_text1,
                token_type_ids_text1=seg_text1_ids,
                input_ids_text2=text2_ids.long(),
                attention_mask_text2=attn_text2,
                token_type_ids_text2=seg_text2_ids,
            )

            avg_val_loss += criterion(logits, labels).item() / len(val_loader)
            valid_preds[idx * args.batch_size : (idx + 1) * args.batch_size] = (
                logits.detach().cpu().squeeze().numpy()
            )
            original[idx * args.batch_size : (idx + 1) * args.batch_size] = (
                labels.detach().cpu().squeeze().numpy()
            )

        score = 0
        preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()

        for i in range(len(args.target_columns)):
            score += np.nan_to_num(target_metric(original[:, i], preds[:, i]))

    return avg_val_loss, score / len(args.target_columns), preds


def infer(args, model, test_loader, test_shape):
    test_preds = np.zeros((test_shape, args.num_classes))
    model.eval()

    for idx, x_batch in enumerate(tqdm(test_loader, desc="Test")):
        #x_feats, d_ids, e_ids, seg_d_ids, seg_e_ids, attn_d, attn_e, target
        with torch.no_grad():
            predictions = model(
                input_ids_text1=x_batch[1].cuda(),
                attention_mask_text1=x_batch[5].cuda(),
                token_type_ids_text1=x_batch[3].cuda(),
                input_ids_text2=x_batch[2].cuda(),
                attention_mask_text2=x_batch[6].cuda(),
                token_type_ids_text2=x_batch[4].cuda()
            )
            test_preds[idx * args.batch_size : (idx + 1) * args.batch_size] = (
                predictions.detach().cpu().squeeze().numpy()
            )

    output = torch.sigmoid(torch.tensor(test_preds)).numpy()
    return output
