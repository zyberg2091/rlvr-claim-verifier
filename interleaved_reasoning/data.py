"""Read and group the original interleaved-reasoning dataset."""

import pandas as pd


def load_data(paths):
    df = pd.read_csv(paths.dataset_csv)

    final_df = df.groupby(['task_id', 'quiz_text']).agg(think_steps=('reasoning_step', lambda x: x.dropna().tolist()),
                                      answers=('sub_answer', lambda x: x.dropna().tolist()),
                                      row_claims=('row_claims', lambda x: x.dropna().tolist()),
                                      final_answer=('final_answer', lambda x: x.dropna().tolist()[0])).reset_index(drop=False)

    AMBIGUOUS_TASKS = ["task_1017871", "task_1034461", "task_119605", "task_142859", "task_225829",
                       "task_271847", "task_358749", "task_370114", "task_41478", "task_434129",
                       "task_460836", "task_544449", "task_557160", "task_563502", "task_693774",
                       "task_810287", "task_810932", "task_812611", "task_822063", "task_839787",
                       "task_858604", "task_900611", "task_931103", "task_949585", "task_991505"]
    final_df = final_df[~final_df['task_id'].isin(AMBIGUOUS_TASKS)]

    return df, final_df
