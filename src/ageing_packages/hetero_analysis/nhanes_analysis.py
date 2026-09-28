"""NHANES loading and grouping used by the manuscript cohort preparation."""

import pandas as pd # type: ignore
import numpy as np # type: ignore
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler # type: ignore

TOPIC_CONFIGS = {
    'diet': {
        'strategy': 'map',
        'topic_column': 'diet',
        'params': {'mapping': {1: 'Good', 3: 'Poor'}},
        'title': 'Survival by Diet Quality (Good vs Poor)',
        'legend_label': 'Diet Quality',
    },
    'income': {
        'strategy': 'quartile',
        'topic_column': 'INDFMMPI',
        'params': {'labels': ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']},
        'title': 'Survival by Income-Poverty Ratio',
        'legend_label': 'Income',
    },
    'alcohol': {
        'strategy': 'bin',
        'topic_column': 'drinks_per_day',
        'params': {
            'bins': [-1, 1, 4, np.inf],
            'labels': ['0-1 drink/day', '2-4 drinks/day', '>4 drinks/day'],
            'right_inclusive': True
        },
        'title': 'Survival by Alcohol Consumption (0-1 vs >4 drinks/day)',
        'legend_label': 'Alcohol Consumption',
    },
    'physical_activity': {
        'strategy': 'custom',
        'params': {'grouper_func': lambda df: _create_activity_groups(df)},
        'title': 'Survival by Physical Activity (No vs Some)',
        'legend_label': 'Physical Activity',
    },
    'sleep_duration': {
        'strategy': 'bin',
        'topic_column': 'sleep_hours',
        'params': {
            'bins': [1, 5, 7, 9, np.inf],
            'labels': ['1-<5 hours', '5-<7 hours', '7-<9 hours', '\u22659 hours'],
            'right_inclusive': False
        },
        'title': 'Survival by Sleep Duration',
        'legend_label': 'Sleep Duration',
    },
    'sleep_frailty': {
        'strategy': 'custom_quartile_extremes',
        'topic_column': 'sleep_frailty',
        'params': {},
        'title': 'Survival by Sleep Frailty (Top vs Bottom Quartile)',
        'legend_label': 'Sleep Frailty',
    },
    'number_of_friends': {
        'strategy': 'bin',
        'topic_column': 'number_of_friends',
        'params': {
            'bins': [-1, 0.1, np.inf],
            'labels': ["0 friends", "1+ friends"],
            'right_inclusive': True
        },
        'title': 'Survival by Number of Friends (0 vs 1+)',
        'legend_label': 'Number of Friends',
    },
    'church_frequency': {
        'strategy': 'bin',
        'topic_column': 'church_frequency',
        'params': {
            'bins': [-1, 0.1, 52, 53],
            'labels': ['never', 'sometimes', 'weekly'],
            'right_inclusive': False
        },
        'title': 'Survival by Church Attendance Frequency',
        'legend_label': 'Church Attendance',
    },
    'education_level': {
        'strategy': 'direct',
        'topic_column': 'education_level',
        'title': 'Survival by Education Level',
        'legend_label': 'Education Level',
    },
}


def load_core(nhanes_data_path):
    # ---- 1. read mortality data ----
    mort = pd.read_csv(os.path.join(nhanes_data_path, "nhanes_mortality_all_years.csv"))
    
    # ---- 2. read age data ----
    age = pd.read_csv(os.path.join(nhanes_data_path, "all_cohort_age_data.csv"),
                      usecols=["SEQN", "age_in_years", "age_at_screening"])
    
    # ---- 3. filter mortality data ----
    mort = mort[mort["eligstat"] == 1]  # keep only linkage-eligible
    
    # ---- 4. merge all data ----
    core = age.merge(mort, on="SEQN")
    
    # ---- 5. construct entry/exit/event ----
    core["entry_age"] = core["age_in_years"].fillna(core["age_at_screening"]).astype(float)
    core["exit_age"] = core["entry_age"] + core["permth_int"] / 12.0
    core["event"] = core["mortstat"]
    return core


def get_topic_df(topic, nhanes_data_path):
    core = load_core(nhanes_data_path)
    if topic == 'diet':
        diet_files = [os.path.join(nhanes_data_path, 'diet', f) for f in os.listdir(os.path.join(nhanes_data_path, 'diet')) if f.startswith('DBQ') and f.endswith('.xpt')]
        df_diet_list = [pd.read_sas(f, format='xport')[['SEQN', 'DBQ700']] for f in diet_files]
        df_diet = pd.concat(df_diet_list, axis=0).reset_index(drop=True)
        def map_diet(x):
            if pd.isna(x) or x in [7, 9]: return np.nan
            elif x in [1, 2]: return 1
            elif x in [3, 4]: return 3
            elif x in [5, 6]: return 2
            else: return np.nan
        df_diet['diet'] = df_diet['DBQ700'].apply(map_diet)
        df_diet = df_diet.drop_duplicates(subset='SEQN', keep='last').dropna(subset=['diet'])
        df_diet['diet'] = df_diet['diet'].round().astype('Int64')
        return core.merge(df_diet[['SEQN', 'diet']], on="SEQN", how="inner")
    elif topic == 'income':
        df_E = pd.read_sas(os.path.join(nhanes_data_path, 'income/INQ_E.xpt'), format='xport')
        df_F = pd.read_sas(os.path.join(nhanes_data_path, 'income/INQ_F.xpt'), format='xport')
        df_G = pd.read_sas(os.path.join(nhanes_data_path, 'income/INQ_G.xpt'), format='xport')
        df_H = pd.read_sas(os.path.join(nhanes_data_path, 'income/INQ_H.xpt'), format='xport')
        df_I = pd.read_sas(os.path.join(nhanes_data_path, 'income/INQ_I.xpt'), format='xport')
        df_J = pd.read_sas(os.path.join(nhanes_data_path, 'income/INQ_J.xpt'), format='xport')
        
        # Convert floats to integers (rounded) while preserving NaN values
        for df in [df_E, df_F, df_G, df_H, df_I, df_J]:
            for col in df.columns:
                if df[col].dtype == 'float64' and col not in ['INDFMMPI', 'INDFMMPC']:
                    df[col] = df[col].round().astype('Int64')  # Use nullable integer type to preserve NaN
        
        # Combine all income dataframes keeping only SEQN and income columns
        df = pd.concat([
            df_E[['SEQN', 'INDFMMPI', 'INDFMMPC']], 
            df_F[['SEQN', 'INDFMMPI', 'INDFMMPC']],
            df_G[['SEQN', 'INDFMMPI', 'INDFMMPC']],
            df_H[['SEQN', 'INDFMMPI', 'INDFMMPC']],
            df_I[['SEQN', 'INDFMMPI', 'INDFMMPC']],
            df_J[['SEQN', 'INDFMMPI']]
        ], axis=0).reset_index(drop=True)
        
        df_income = df.drop_duplicates(subset='SEQN', keep='last').dropna(subset=['INDFMMPI'])
        return core.merge(df_income[['SEQN', 'INDFMMPI']], on="SEQN", how="inner")
    elif topic == 'physical_activity':
        # Load only the specified files
        paq_files = ['PAQ_E.xpt', 'PAQ_F.xpt', 'PAQ_G.xpt', 'PAQ_H.xpt', 'PAQ_I.xpt', 'PAQ_J.xpt']
        paq_paths = [os.path.join(nhanes_data_path, 'physical_activity', f) for f in paq_files]
        df_pa_list = [pd.read_sas(f, format='xport') for f in paq_paths]
        # Round float columns to Int64
        for df in df_pa_list:
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = df[col].round().astype('Int64')
        df_activity = pd.concat(df_pa_list, axis=0).reset_index(drop=True)
        # Clean activity variables
        def clean_activity_variable(df, col_name, default_value=0):
            if col_name in df.columns:
                df[col_name] = df[col_name].replace([7, 77, 7777, 9, 99, 9999], np.nan)
                if 'PAQ' in col_name:
                    df[col_name] = df[col_name].fillna(default_value)
            return df
        activity_vars = ['PAQ605', 'PAQ610', 'PAD615', 'PAQ620', 'PAQ625', 'PAD630',
                        'PAQ635', 'PAQ640', 'PAD645', 'PAQ650', 'PAQ655', 'PAD660',
                        'PAQ665', 'PAQ670', 'PAD675']
        for var in activity_vars:
            df_activity = clean_activity_variable(df_activity, var)
        # Calculate MET-minutes per week
        df_activity['vigorous_work_mets'] = 0.0
        df_activity['moderate_work_mets'] = 0.0
        df_activity['transport_mets'] = 0.0
        df_activity['vigorous_rec_mets'] = 0.0
        df_activity['moderate_rec_mets'] = 0.0
        # 1. Vigorous Work
        vig_work_cond = (df_activity.get('PAQ605') == 1) & df_activity.get('PAQ610', pd.Series(0)).notna() & df_activity.get('PAD615', pd.Series(0)).notna()
        df_activity.loc[vig_work_cond, 'vigorous_work_mets'] = df_activity.loc[vig_work_cond, 'PAQ610'] * df_activity.loc[vig_work_cond, 'PAD615'] * 8.0
        # 2. Moderate Work
        mod_work_cond = (df_activity.get('PAQ620') == 1) & df_activity.get('PAQ625', pd.Series(0)).notna() & df_activity.get('PAD630', pd.Series(0)).notna()
        df_activity.loc[mod_work_cond, 'moderate_work_mets'] = df_activity.loc[mod_work_cond, 'PAQ625'] * df_activity.loc[mod_work_cond, 'PAD630'] * 4.0
        # 3. Transport
        transport_cond = (df_activity.get('PAQ635') == 1) & df_activity.get('PAQ640', pd.Series(0)).notna() & df_activity.get('PAD645', pd.Series(0)).notna()
        df_activity.loc[transport_cond, 'transport_mets'] = df_activity.loc[transport_cond, 'PAQ640'] * df_activity.loc[transport_cond, 'PAD645'] * 4.0
        # 4. Vigorous Rec
        vig_rec_cond = (df_activity.get('PAQ650') == 1) & df_activity.get('PAQ655', pd.Series(0)).notna() & df_activity.get('PAD660', pd.Series(0)).notna()
        df_activity.loc[vig_rec_cond, 'vigorous_rec_mets'] = df_activity.loc[vig_rec_cond, 'PAQ655'] * df_activity.loc[vig_rec_cond, 'PAD660'] * 8.0
        # 5. Moderate Rec
        mod_rec_cond = (df_activity.get('PAQ665') == 1) & df_activity.get('PAQ670', pd.Series(0)).notna() & df_activity.get('PAD675', pd.Series(0)).notna()
        df_activity.loc[mod_rec_cond, 'moderate_rec_mets'] = df_activity.loc[mod_rec_cond, 'PAQ670'] * df_activity.loc[mod_rec_cond, 'PAD675'] * 4.0
        # Fill NaN MET columns with 0
        met_columns = ['vigorous_work_mets', 'moderate_work_mets', 'transport_mets', 'vigorous_rec_mets', 'moderate_rec_mets']
        for col in met_columns:
            if col not in df_activity.columns:
                df_activity[col] = 0.0
            else:
                df_activity[col] = df_activity[col].fillna(0.0)
        df_activity['total_met_minutes_week'] = (
            df_activity['vigorous_work_mets'] +
            df_activity['moderate_work_mets'] +
            df_activity['transport_mets'] +
            df_activity['vigorous_rec_mets'] +
            df_activity['moderate_rec_mets']
        )
        # Log transform and min-max scale
        df_activity['log_total_mets'] = np.log(df_activity['total_met_minutes_week'] + 1)
        scaler = MinMaxScaler()
        df_activity['physical_activity_index'] = scaler.fit_transform(df_activity[['log_total_mets']]).flatten()
        # Merge with core and drop missing
        df_activity_merged = core.merge(df_activity[['SEQN', 'physical_activity_index']], on="SEQN", how="inner")
        df_activity_merged = df_activity_merged.dropna(subset=["physical_activity_index", "entry_age", "exit_age"])
        return df_activity_merged
    elif topic == 'sleep_duration':
        sleep_files = [os.path.join(nhanes_data_path, 'sleep', f) for f in os.listdir(os.path.join(nhanes_data_path, 'sleep')) if f.startswith('SLQ') and f.endswith('.xpt')]
        df_sleep_list = []
        for f in sleep_files:
            df_temp = pd.read_sas(f, format='xport')
            if 'SLD010H' in df_temp.columns: df_temp = df_temp.rename(columns={'SLD010H': 'sleep_hours'})
            elif 'SLD012' in df_temp.columns: df_temp = df_temp.rename(columns={'SLD012': 'sleep_hours'})
            df_sleep_list.append(df_temp)
        df_sleep_all_years = pd.concat(df_sleep_list, axis=0, ignore_index=True)
        df_sleep_all_years = df_sleep_all_years.drop_duplicates(subset='SEQN', keep='last').dropna(subset=['sleep_hours'])
        df_sleep_all_years['sleep_hours'] = df_sleep_all_years['sleep_hours'].round().astype('Int64')
        return core.merge(df_sleep_all_years[['SEQN', 'sleep_hours']], on="SEQN", how="inner")
    elif topic == 'sleep_frailty':
        df_sleep_d = pd.read_sas(os.path.join(nhanes_data_path, 'sleep/SLQ_D.xpt'), format='xport')
        df_sleep_e = pd.read_sas(os.path.join(nhanes_data_path, 'sleep/SLQ_E.xpt'), format='xport')
        # Round float columns to Int64 to preserve NaN
        for df in [df_sleep_d, df_sleep_e]:
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = df[col].round().astype('Int64')
        df_sleep_frailty = pd.concat([df_sleep_d, df_sleep_e], axis=0, ignore_index=True).copy()
        def fill_vals(df, column_name, exclude_values):
            if column_name not in df.columns: return df
            valid_data = df[~df[column_name].isin(exclude_values)]
            median_value = valid_data[column_name].median()
            df.loc[:, column_name] = df[column_name].replace({val: median_value for val in exclude_values})
            return df
        for col in ['SLQ050', 'SLQ060', 'SLQ070A', 'SLQ070B', 'SLQ070C', 'SLQ070D']:
            if col in df_sleep_frailty.columns: df_sleep_frailty.loc[:, col] = df_sleep_frailty[col].fillna(0)
        for col in ['SLD010H', 'SLD020M']: df_sleep_frailty = fill_vals(df_sleep_frailty, col, [77, 99])
        for col in ['SLQ030', 'SLQ040', 'SLQ050', 'SLQ060', 'SLQ070A', 'SLQ080', 'SLQ090', 'SLQ100', 'SLQ110', 'SLQ120', 'SLQ130', 'SLQ140', 'SLQ150', 'SLQ160', 'SLQ170', 'SLQ180', 'SLQ190', 'SLQ200', 'SLQ210', 'SLQ220', 'SLQ230', 'SLQ240']:
            df_sleep_frailty = fill_vals(df_sleep_frailty, col, [7, 9])
        columns_to_scale = [col for col in df_sleep_frailty.columns if ('SLQ' in col or 'SLD' in col) and col != 'SEQN']
        # Standardize SLD010H as in notebook
        if 'SLD010H' in df_sleep_frailty.columns:
            scaler_std = StandardScaler()
            df_sleep_frailty['SLD010H'] = abs(scaler_std.fit_transform(df_sleep_frailty[['SLD010H']]))
        # MinMax scale all SLQ/SLD columns
        scaler = MinMaxScaler()
        df_sleep_frailty[columns_to_scale] = scaler.fit_transform(df_sleep_frailty[columns_to_scale])
        df_sleep_frailty['sleep_frailty'] = df_sleep_frailty[columns_to_scale].mean(axis=1)
        df_sleep_frailty = df_sleep_frailty.drop_duplicates(subset='SEQN', keep='last').dropna(subset=['sleep_frailty'])
        return core.merge(df_sleep_frailty[['SEQN', 'sleep_frailty']], on="SEQN", how="inner")
    elif topic == 'alcohol':
        alc_files = [os.path.join(nhanes_data_path, 'alcohol', f) for f in os.listdir(os.path.join(nhanes_data_path, 'alcohol')) if f.startswith('ALQ') and f.endswith('.xpt')]
        df_alc_list = [pd.read_sas(f, format='xport')[['SEQN', 'ALQ130']] for f in alc_files]
        df_alcohol = pd.concat(df_alc_list, axis=0).rename(columns={'ALQ130': 'drinks_per_day'}).reset_index(drop=True)
        df_alcohol = df_alcohol[~df_alcohol['drinks_per_day'].isin([777, 999, 77, 99])]
        df_alcohol = df_alcohol.drop_duplicates(subset='SEQN', keep='last').dropna(subset=['drinks_per_day'])
        df_alcohol['drinks_per_day'] = df_alcohol['drinks_per_day'].round().astype('Int64')
        return core.merge(df_alcohol, on="SEQN", how="inner")
    elif topic == 'number_of_friends':
        ss_files_map = {'SSQ.xpt': {'need': 'SSQ030', 'friends': 'SSQ060'}, 'SSQ_B.xpt': {'need': 'SSD031', 'friends': 'SSD061'}, 'SSQ_C.xpt': {'need': 'SSQ031', 'friends': 'SSQ061'}, 'SSQ_D.xpt': {'need': 'SSQ031', 'friends': 'SSQ061'}, 'SSQ_E.xpt': {'need': 'SSQ031', 'friends': 'SSQ061'}}
        df_ss_list = []
        for f, cols in ss_files_map.items():
            df_temp = pd.read_sas(os.path.join(nhanes_data_path, 'social_support', f), format='xport')
            df_ss_list.append(df_temp[['SEQN', cols['need'], cols['friends']]].rename(columns={cols['need']: 'need_support', cols['friends']: 'number_of_friends'}))
        df_social_support = pd.concat(df_ss_list, axis=0).reset_index(drop=True)
        # SSQ close-friend counts 7 and 9 are valid responses.
        invalid_codes = [7777, 9999]
        df_social_support['number_of_friends'] = df_social_support['number_of_friends'].replace(invalid_codes, np.nan)
        df_social_support['need_support'] = df_social_support['need_support'].replace(invalid_codes, np.nan)
        df_social_support = df_social_support.drop_duplicates(subset='SEQN', keep='last')
        df_social_support = df_social_support.dropna(subset=['number_of_friends'])
        df_social_support['number_of_friends'] = df_social_support['number_of_friends'].round().astype('Int64')
        return core.merge(df_social_support[['SEQN', 'number_of_friends']], on="SEQN", how="inner")
    elif topic == 'church_frequency':
        # Load SSQ_D and SSQ_E, extract SSD044, clean, and merge
        ssq_files = [os.path.join(nhanes_data_path, 'social_support', f) for f in ['SSQ_D.xpt', 'SSQ_E.xpt']]
        df_church_list = []
        for f in ssq_files:
            df_temp = pd.read_sas(f, format='xport')
            if 'SSD044' in df_temp.columns:
                df_church_list.append(df_temp[['SEQN', 'SSD044']].rename(columns={'SSD044': 'church_frequency'}))
        if not df_church_list:
            raise ValueError('No church frequency data found in SSQ_D/E')
        df_church = pd.concat(df_church_list, axis=0).reset_index(drop=True)
        df_church['church_frequency'] = df_church['church_frequency'].replace([77777, 99999], np.nan)
        # Round to nearest integer
        df_church['church_frequency'] = df_church['church_frequency'].round().astype('Int64')
        df_church = df_church.drop_duplicates(subset='SEQN', keep='last').dropna(subset=['church_frequency'])
        return core.merge(df_church[['SEQN', 'church_frequency']], on="SEQN", how="inner")
    elif topic == 'education_level':
        demo_dir = os.path.join(nhanes_data_path, 'demo')
        demo_files = [os.path.join(demo_dir, f) for f in os.listdir(demo_dir) if f.startswith('DEMO') and f.endswith('.xpt')]
        df_demo_list = []
        for f in demo_files:
            df_temp = pd.read_sas(f, format='xport')
            if 'DMDEDUC2' in df_temp.columns:
                df_demo_list.append(df_temp[['SEQN', 'DMDEDUC2']])
        if not df_demo_list:
            raise ValueError('No education data found in DEMO files')
        df_educ = pd.concat(df_demo_list, axis=0).reset_index(drop=True)
        def map_educ(x):
            if pd.isna(x) or x in [7, 9]:
                return np.nan
            elif x in [1, 2]:
                return 'no highschool'
            elif x == 3:
                return 'high school'
            elif x in [4, 5]:
                return 'some college'
            else:
                return np.nan
        df_educ['education_level'] = df_educ['DMDEDUC2'].apply(map_educ)
        # Ensure only valid categories or NaN
        valid_educ = ['no highschool', 'high school', 'some college']
        df_educ.loc[~df_educ['education_level'].isin(valid_educ), 'education_level'] = np.nan
        df_educ = df_educ.drop_duplicates(subset='SEQN', keep='last').dropna(subset=['education_level'])
        return core.merge(df_educ[['SEQN', 'education_level']], on="SEQN", how="inner")
    else:
        raise ValueError(f"Unknown topic: {topic}")


def _create_activity_groups(df):
    """Custom grouping function for physical activity: No Activity vs Some Activity."""
    df_copy = df.copy()
    group_col = 'activity_group'
    df_copy[group_col] = pd.NA
    
    # No Activity (physical_activity_index == 0)
    no_activity_mask = df_copy['physical_activity_index'] == 0
    df_copy.loc[no_activity_mask, group_col] = 'No Activity'
    
    # Some Activity (physical_activity_index > 0)
    some_activity_mask = df_copy['physical_activity_index'] > 0
    df_copy.loc[some_activity_mask, group_col] = 'Some Activity'
    
    return df_copy, group_col


def _apply_grouping_strategy(df, config):
    """Applies a grouping strategy from the config to a DataFrame."""
    df_copy = df.copy()
    strategy = config['strategy']
    params = config.get('params', {})

    if strategy == 'direct':
        return df_copy, config['topic_column']
    if strategy == 'custom':
        return params['grouper_func'](df_copy)
    if strategy == 'custom_quartile_extremes':
        # For sleep_frailty: create quartiles, keep only 0 and 3
        topic_col = config['topic_column']
        quartile_col = f"{topic_col}_quartile"
        df_copy = df_copy.dropna(subset=[topic_col])
        df_copy[quartile_col] = pd.qcut(df_copy[topic_col], q=4, labels=False, duplicates='drop')
        # Only keep lowest and highest quartiles
        mask = df_copy[quartile_col].isin([0, 3])
        df_copy = df_copy[mask].copy()
        label_map = {0: 'Q1 (lowest)', 3: 'Q4 (highest)'}
        df_copy[quartile_col] = df_copy[quartile_col].map(label_map)
        return df_copy, quartile_col
    
    topic_col = config['topic_column']
    group_col = f"{topic_col}_group"

    if strategy == 'quartile':
        labels = params.get('labels') or ['Q1', 'Q2', 'Q3', 'Q4']
        df_copy[group_col] = pd.qcut(df_copy[topic_col], q=4, labels=labels, duplicates='drop')
    elif strategy == 'map':
        df_copy[group_col] = df_copy[topic_col].map(params['mapping'])
    elif strategy == 'bin':
        df_copy[group_col] = pd.cut(df_copy[topic_col], bins=params['bins'], labels=params['labels'], right=params.get('right_inclusive', True), include_lowest=False)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return df_copy, group_col
