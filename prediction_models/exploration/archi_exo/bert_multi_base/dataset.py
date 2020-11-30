from math import ceil, floor

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from text_model_boris.utils import misc


def _get_masks(tokens: list, tokenizer, max_seq_length: int) -> list:
    """Mask for padding"""
    tokens = tokens[:max_seq_length]
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens: list, tokenizer, max_seq_length: int) -> list:
    """Segments: 0 for the first sequence, 1 for the second"""
    tokens = tokens[:max_seq_length]

    segments = []
    first_sep = True
    current_segment_id = 0

    for token in tokens:
        segments.append(current_segment_id)
        if token == tokenizer.sep_token:
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def _get_ids(tokens: list, tokenizer, max_seq_length: list) -> list:
    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = token_ids[:max_seq_length]
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def _trim_input(
    args,
    tokenizer,
    short_text,
    long_text1,
    long_text2,
    max_sequence_length=290,
    st_max_len=30,
    lt1_max_len=128,
    lt2_max_len=128,
):

    st = tokenizer.tokenize(short_text)
    lt1 = tokenizer.tokenize(long_text1)
    lt2 = tokenizer.tokenize(long_text2)

    st_len = len(st)
    lt1_len = len(lt1)
    lt2_len = len(lt2)

    if (st_len + lt1_len + lt2_len + 4) > max_sequence_length:

        if st_max_len > st_len:
            st_new_len = st_len
            lt2_max_len = lt2_max_len + floor((st_max_len - st_len) / 2)
            lt1_max_len = lt1_max_len + ceil((st_max_len - st_len) / 2)
        else:
            st_new_len = st_max_len

        if lt2_max_len > lt2_len:
            lt2_new_len = lt2_len
            lt1_new_len = lt1_max_len + (lt2_max_len - lt2_len)
        elif lt1_max_len > lt1_len:
            lt2_new_len = lt2_max_len + (lt1_max_len - lt1_len)
            lt1_new_len = lt1_len
        else:
            lt2_new_len = lt2_max_len
            lt1_new_len = lt1_max_len

        if st_new_len + lt1_new_len + lt2_new_len + 4 != max_sequence_length:
            raise ValueError(
                "New sequence length should be %d, but is %d"
                % (max_sequence_length, (st_new_len + lt1_new_len + lt2_new_len + 4))
            )
        lt1_len_head = round(lt1_new_len / 2)
        lt1_len_tail = -1 * (lt1_new_len - lt1_len_head)
        lt2_len_head = round(lt2_new_len / 2)
        lt2_len_tail = -1 * (lt2_new_len - lt2_len_head)  # Head+Tail method
        st = st[:st_new_len]
        if args.head_tail:
            lt1 = lt1[:lt1_len_head] + lt1[lt1_len_tail:]
            lt2 = lt2[:lt2_len_head] + lt2[lt2_len_tail:]
        else:
            lt1 = lt1[:lt1_new_len]
            lt2 = lt2[:lt2_new_len]  # No Head+Tail ,usual processing

    return st, lt1, lt2


def _convert_to_bert_inputs(
    short_text: str, long_text1: str, long_text2: str, tokenizer, max_sequence_length: int, model_type: str,
) -> list:
    """Converts tokenized input to ids, masks and segments for BERT"""
    if model_type == "roberta":
        stoken = (
            [tokenizer.cls_token]
            + short_text
            + [tokenizer.sep_token] * 2
            + long_text1
            + [tokenizer.sep_token] * 2
            + long_text2
            + [tokenizer.sep_token] * 2
        )

    else:
        stoken = (
            [tokenizer.cls_token]
            + short_text
            + [tokenizer.sep_token]
            + long_text1
            + [tokenizer.sep_token]
            + long_text2
            + [tokenizer.sep_token]
        )

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, tokenizer, max_sequence_length)
    input_segments = _get_segments(stoken, tokenizer, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arrays(
    args,
    df,
    columns: list,
    tokenizer,
    max_sequence_length: int,
    st_max_len: int = 30,
    lt1_max_len: int = 128,
    lt2_max_len: int = 128,
    verbose: bool = False,
):
    input_ids, input_masks, input_segments = [], [], []

    iterator = df[columns].iterrows()
    if verbose:
        iterator = tqdm(iterator, desc="Preparing dataset",
                        total=len(df))

    for _, instance in iterator:
        st, lt1, lt2 = (
            instance[args.input_columns[0]],
            instance[args.input_columns[1]],
            instance[args.input_columns[2]],
        )
        st, lt1, lt2 = _trim_input(
            args,
            tokenizer,
            st,
            lt1,
            lt2,
            max_sequence_length,
            st_max_len,
            lt1_max_len,
            lt2_max_len,
        )
        ids, masks, segments = _convert_to_bert_inputs(
            st, lt1, lt2, tokenizer, max_sequence_length, args.model_type,
        )

        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return (
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    )


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        df,
        tokenizer,
        test=False,
        short_text_transform=None,
        body_transform=None,
        long_text2_transform=None,
    ):
        self.data = df
        self.tokenizer = tokenizer
        self.is_test = test
        self.params = args

        self.short_text_transform = short_text_transform
        self.body_transform = body_transform
        self.long_text2_transform = long_text2_transform

    @classmethod
    def from_frame(
        cls,
        args,
        df,
        tokenizer,
        test=False,
        short_text_transform=None,
        body_transform=None,
        long_text2_transform=None,
    ):
        return cls(
            args,
            df,
            tokenizer,
            test=test,
            short_text_transform=short_text_transform,
            body_transform=body_transform,
            long_text2_transform=long_text2_transform,
        )

    def __getitem__(self, idx):
        instance = self.data[idx: idx + 1].copy()

        for col, transform in zip(
            misc.input_columns,
            [self.short_text_transform, self.body_transform,
                self.long_text2_transform],
        ):
            if transform is not None:
                instance[col] = transform[col]

        input_ids, input_masks, input_segments = compute_input_arrays(
            self.params,
            instance,
            self.params.input_columns,
            self.tokenizer,
            max_sequence_length=self.params.max_sequence_length,
            st_max_len=self.params.max_short_text_length,
            lt1_max_len=self.params.max_long_text1_length,
            lt2_max_len=self.params.max_long_text2_length,
            verbose=False,
        )
        length = torch.sum(input_masks != 0)
        input_ids, input_masks, input_segments = (
            torch.squeeze(input_ids, dim=0),
            torch.squeeze(input_masks, dim=0),
            torch.squeeze(input_segments, dim=0),
        )

        if not self.is_test:
            labels = compute_output_arrays(
                instance, self.params.target_columns)
            labels = torch.tensor(labels, dtype=torch.float32)
            labels = torch.squeeze(labels, dim=0)
            return input_ids, input_masks, input_segments, labels, length
        else:
            return input_ids, input_masks, input_segments, length

    def __len__(self):
        return len(self.data)


def cross_validation_split(args, train_df, tokenizer, ignore_train=False):
    kf = StratifiedKFold(n_splits=args.folds)
    train_df_y = train_df[args.target_columns].values.argmax(axis=1)

    for fold, (train_index, val_index) in enumerate(
        kf.split(train_df.values, y=train_df_y)
    ):

        if args.use_folds is not None and fold not in args.use_folds:
            continue

        if not ignore_train:
            train_subdf = train_df.iloc[train_index]
            train_set = TextDataset.from_frame(args, train_subdf, tokenizer)
        else:
            train_set = None

        valid_set = TextDataset.from_frame(
            args, train_df.iloc[val_index], tokenizer)

        yield (
            fold,
            train_set,
            valid_set,
            train_df.iloc[train_index],
            train_df.iloc[val_index],
        )


def get_pseudo_set(args, pseudo_df, tokenizer):
    return TextDataset.from_frame(args, pseudo_df, tokenizer)


def get_test_set(args, test_df, tokenizer):
    return TextDataset.from_frame(args, test_df, tokenizer, True)
