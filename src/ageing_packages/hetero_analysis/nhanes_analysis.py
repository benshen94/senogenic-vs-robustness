import pandas as pd # type: ignore
import numpy as np # type: ignore
import os
from lifelines import KaplanMeierFitter # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler # type: ignore
import warnings
import matplotlib.colors as mcolors # type: ignore
from adjustText import adjust_text # type: ignore
from pathlib import Path
from ageing_packages.mortality_models.gamma_gompertz import GammaGompertz # type: ignore
from ageing_packages.utils import sr_utils as sr_utils # type: ignore
import pickle # type: ignore
import matplotlib.lines as mlines # type: ignore
import matplotlib.patches as mpatches # type: ignore
# Suppress lifelines approximation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lifelines')
warnings.filterwarnings('ignore', message='.*Approximating using.*survival_function_.*')

# =============================================================================
# Configuration
# =============================================================================

# Centralized configuration for all analysis topics
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
    'work_regularity': {
        'strategy': 'direct',
        'topic_column': 'work_schedule',
        'title': 'Survival by Work Schedule',
        'legend_label': 'Work Schedule',
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
nhanes_data_path = str(PROJECT_ROOT / "data" / "nhanes") + os.sep
results_path = str(PROJECT_ROOT / "results") + os.sep
DEFAULT_SR_USA_2019_PKL = os.path.join(results_path, 'param_variation_results_usa_2019.pkl')
DEFAULT_SR_USA_2019_UNDERLAY_CACHE = os.path.join(results_path, 'sr_usa_2019_steepness_longevity_curves.pkl')
DEFAULT_SR_PARAM_ORDER = ('eta', 'beta', 'kappa', 'epsilon', 'Xc')

# =============================================================================
# Data Loading Functions
# =============================================================================

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

def plot_baseline_km(ax, mgg_fit=False, zero_m = True):
    """
    Plot the Kaplan-Meier curve for the baseline (all cohort) in black.
    
    Args:
        ax: matplotlib axes object to plot on
        without_extrinsic (bool): If True, uses MGG model with m=0 (no extrinsic mortality)
                                 If False, uses actual NHANES data (default)
    """
    # Load core data first
    core = load_core(nhanes_data_path)
    
    # Create KMF object and fit to actual NHANES data
    kmf = KaplanMeierFitter()
    
    # Check if required columns exist before fitting
    required_cols = ['exit_age', 'event', 'entry_age']
    missing_cols = [col for col in required_cols if col not in core.columns]
    
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print(f"Available columns: {core.columns.tolist()}")
        return None
    
    # Fit the baseline curve for all participants
    kmf.fit(
        durations=core["exit_age"],
        event_observed=core["event"],
        entry=core["entry_age"],
        label="Baseline (All)",
    )
    
    if mgg_fit:
        # Use MGG model fitted to the KMF data with m=0 (no extrinsic mortality)
        try:
            # Fit MGG model to the KMF data
            mgg = GammaGompertz()
            mgg.fit_params(kmf=kmf, print_out=True)
            
            # Set m=0 to remove extrinsic mortality
            if zero_m:
                mgg.m = 0
            
            death_times = mgg.sample_death_times(n=100000, min_age=core['entry_age'].min())
            kmf2 = KaplanMeierFitter()
            kmf2.fit(death_times, event_observed=np.ones(len(death_times)))
            kmf2.plot_survival_function(ax=ax, color='blue', linewidth=2, label="Baseline (No Extrinsic)")
            
        except Exception as e:
            print(f"Warning: Could not plot baseline without extrinsic mortality: {e}")
            print("Falling back to baseline with extrinsic mortality...")
            without_extrinsic = False
    
    if not mgg_fit:
        # Plot the actual NHANES KMF curve in black
        kmf.plot_survival_function(ax=ax, color='black', linewidth=2)
    
    return kmf

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
        invalid_codes = [7, 9, 77, 99]
        df_social_support['number_of_friends'] = df_social_support['number_of_friends'].replace(invalid_codes, np.nan)
        df_social_support['need_support'] = df_social_support['need_support'].replace(invalid_codes, np.nan)
        df_social_support = df_social_support.drop_duplicates(subset='SEQN', keep='last')
        df_social_support = df_social_support.dropna(subset=['number_of_friends'])
        df_social_support['number_of_friends'] = df_social_support['number_of_friends'].round().astype('Int64')
        return core.merge(df_social_support[['SEQN', 'number_of_friends']], on="SEQN", how="inner")
    elif topic == 'work_regularity':
        work_files = [os.path.join(nhanes_data_path, 'occupation', f) for f in os.listdir(os.path.join(nhanes_data_path, 'occupation')) if f.startswith('OCQ') and f.endswith('.xpt')]
        df_work_list = []
        for f in work_files:
            df_temp = pd.read_sas(f, format='xport')
            if 'OCQ265' in df_temp.columns: df_work_list.append(df_temp[['SEQN', 'OCQ265']])
        df_work = pd.concat(df_work_list, axis=0).rename(columns={'OCQ265': 'work_hours'}).reset_index(drop=True)
        def categorize_work_schedule(work_hours):
            if work_hours == 1: return "Daytime work"
            elif work_hours in [2, 3]: return "Night shift"
            elif work_hours in [4, 5]: return "Rotating shift"
            else: return None
        df_work['work_schedule'] = df_work['work_hours'].apply(categorize_work_schedule)
        df_work = df_work.drop_duplicates(subset='SEQN', keep='last').dropna(subset=['work_schedule'])
        return core.merge(df_work[['SEQN', 'work_hours', 'work_schedule']], on="SEQN", how="inner")
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

# =============================================================================
# Grouping Logic
# =============================================================================

def get_expected_group_names(topic_name):
    """
    Get the expected group names for a topic from TOPIC_CONFIGS.
    This ensures consistency between data generation and plotting.
    
    Args:
        topic_name (str): Name of the topic
        
    Returns:
        list: List of expected group names for this topic
    """
    if topic_name not in TOPIC_CONFIGS:
        raise ValueError(f"Topic '{topic_name}' not found in TOPIC_CONFIGS")
    
    config = TOPIC_CONFIGS[topic_name]
    strategy = config['strategy']
    params = config.get('params', {})
    
    if strategy == 'direct':
        # For direct strategy, we need to check the actual data to know possible values
        # This is handled case-by-case below
        if topic_name == 'education_level':
            return ['no highschool', 'high school', 'some college']
        elif topic_name == 'work_regularity':
            return ['Daytime work', 'Night shift', 'Rotating shift']
        else:
            return []  # Unknown direct strategy topic
    
    elif strategy == 'custom':
        # Custom strategies are handled case-by-case
        if topic_name == 'physical_activity':
            return ['No Activity', 'Some Activity']
        else:
            return []  # Unknown custom strategy topic
    
    elif strategy == 'custom_quartile_extremes':
        # For quartile extremes, always these two groups
        return ['Q1 (lowest)', 'Q4 (highest)']
    
    elif strategy == 'map':
        # For mapping strategy, return the mapped values
        mapping = params.get('mapping', {})
        return list(mapping.values())
    
    elif strategy in ['quartile', 'bin']:
        # For quartile and bin strategies, return the labels
        return params.get('labels', [])
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def validate_topic_consistency(interventions_dict, verbose=True):
    """
    Validate that all topics in interventions_dict have consistent group names
    with what's expected from TOPIC_CONFIGS.
    
    Args:
        interventions_dict (dict): The interventions dictionary
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Dictionary with validation results for each topic
    """
    validation_results = {}
    
    for topic_name in TOPIC_CONFIGS.keys():
        if topic_name not in interventions_dict:
            if verbose:
                print(f"⚠️  Topic '{topic_name}' missing from interventions_dict")
            validation_results[topic_name] = {'status': 'missing', 'details': 'Topic not found in interventions_dict'}
            continue
        
        try:
            expected_groups = get_expected_group_names(topic_name)
            actual_groups = list(interventions_dict[topic_name].keys())
            
            missing_groups = [g for g in expected_groups if g not in actual_groups]
            extra_groups = [g for g in actual_groups if g not in expected_groups]
            
            if not missing_groups and not extra_groups:
                if verbose:
                    print(f"✅ Topic '{topic_name}': All groups match")
                validation_results[topic_name] = {'status': 'perfect_match'}
            else:
                if verbose:
                    print(f"⚠️  Topic '{topic_name}': Mismatch detected")
                    print(f"    Expected: {expected_groups}")
                    print(f"    Actual: {actual_groups}")
                    if missing_groups:
                        print(f"    Missing: {missing_groups}")
                    if extra_groups:
                        print(f"    Extra: {extra_groups}")
                
                validation_results[topic_name] = {
                    'status': 'mismatch',
                    'expected': expected_groups,
                    'actual': actual_groups,
                    'missing': missing_groups,
                    'extra': extra_groups
                }
        
        except Exception as e:
            if verbose:
                print(f"❌ Topic '{topic_name}': Validation failed - {e}")
            validation_results[topic_name] = {'status': 'error', 'details': str(e)}
    
    return validation_results

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

# =============================================================================
# Core Analysis Functions
# =============================================================================

def calculate_steepness_from_kmf(kmf, min_age, ages_for_pdf):
    """
    Calculate steepness metrics from Kaplan-Meier survival curve.
    
    This function properly calculates steepness using the survival curve rather than
    death times percentiles. For IQR steepness, it uses kmf.percentile() to get
    survival percentiles, and for CV steepness, it calculates the PDF from the
    survival curve and computes CV from the PDF.
    
    Args:
        kmf: Fitted KaplanMeierFitter object
        min_age: Minimum entry age for relative calculations
        ages_for_pdf: Array of ages for PDF calculation
        
    Returns:
        dict: Dictionary with steepness metrics including:
            - steepness_iqr_absolute: IQR steepness using absolute times
            - steepness_iqr_relative: IQR steepness using relative times (t - min_age)
            - steepness_cv_absolute: CV steepness calculated from PDF using absolute times
            - steepness_cv_relative: CV steepness calculated from PDF using relative times
            - pdf: Probability density function derived from survival curve
            - ages: Age array corresponding to PDF
    """
    # Get survival percentiles from KM curve
    t25 = kmf.percentile(0.25)
    t50 = kmf.percentile(0.50)
    t75 = kmf.percentile(0.75)
    
    # Calculate IQR steepness (absolute)
    steepness_iqr_absolute = None
    if t25 is not None and t50 is not None and t75 is not None and t75 != t25:
        steepness_iqr_absolute = abs(-t50 / (t75 - t25))
    
    # Calculate IQR steepness (relative) - subtract min_age after getting percentiles
    steepness_iqr_relative = None
    if t25 is not None and t50 is not None and t75 is not None and t75 != t25:
        t25_relative = t25 - min_age
        t50_relative = t50 - min_age
        t75_relative = t75 - min_age
        if t75_relative != t25_relative:
            steepness_iqr_relative = abs(-t50_relative / (t75_relative - t25_relative))
    
    # Calculate CV steepness from PDF
    # Interpolate survival function to regular age grid
    survival_prob_interp = np.interp(ages_for_pdf, kmf.timeline, kmf.survival_function_.values.flatten(), left=1.0, right=0.0)
    
    # Calculate PDF as negative derivative of survival function
    pdf = -np.diff(survival_prob_interp, prepend=1.0)
    
    # Normalize PDF to sum to 1
    pdf = pdf / np.sum(pdf)
    
    # Calculate CV from PDF
    steepness_cv_absolute = None
    steepness_cv_relative = None
    
    if np.sum(pdf) > 0:
        # For absolute CV: use ages_for_pdf as time values
        mean_time_absolute = np.sum(ages_for_pdf * pdf)
        var_time_absolute = np.sum((ages_for_pdf - mean_time_absolute)**2 * pdf)
        std_time_absolute = np.sqrt(var_time_absolute)
        
        if mean_time_absolute > 0 and std_time_absolute > 0:
            cv_absolute = std_time_absolute / mean_time_absolute
            steepness_cv_absolute = 1 / cv_absolute
        
        # For relative CV: use (ages_for_pdf - min_age) as time values
        ages_relative = ages_for_pdf - min_age
        mean_time_relative = np.sum(ages_relative * pdf)
        var_time_relative = np.sum((ages_relative - mean_time_relative)**2 * pdf)
        std_time_relative = np.sqrt(var_time_relative)
        
        if mean_time_relative > 0 and std_time_relative > 0:
            cv_relative = std_time_relative / mean_time_relative
            steepness_cv_relative = 1 / cv_relative
    
    return {
        'steepness_iqr_absolute': steepness_iqr_absolute,
        'steepness_iqr_relative': steepness_iqr_relative,
        'steepness_cv_absolute': steepness_cv_absolute,
        'steepness_cv_relative': steepness_cv_relative,
        'pdf': pdf,
        'ages': ages_for_pdf
    }

def safe_nanstd(arr):
    arr = [x for x in arr if x is not None]
    if len(arr) == 0:
        return np.nan
    return np.nanstd(arr)

def calculate_survival_stats(nhanes_data_path, configs=TOPIC_CONFIGS, n_bootstrap=100, print_bootstrap=False):
    """Calculates and stores survival statistics for all configured topics using per-topic DataFrames, with bootstrapped errorbars for steepness and median lifespan.
    If print_bootstrap is True, prints every bootstrap iteration for each group. If False, prints only topic/group progress and summary stats."""
    results = {}
    kmf = KaplanMeierFitter()
    ages_for_pdf = np.arange(20, 111, 1)

    # Baseline: use all eligible participants
    print("Processing baseline (all participants)...")
    df_baseline = load_core(nhanes_data_path)
    baseline_required_cols = ['entry_age', 'exit_age', 'event']
    df_baseline = df_baseline.dropna(subset=baseline_required_cols)
    
    if not df_baseline.empty:
        results['baseline'] = {'with_extrinsic': {}, 'without_extrinsic': {}}
        
        # Calculate baseline with extrinsic mortality (current approach)
        kmf.fit(df_baseline['exit_age'], event_observed=df_baseline['event'], entry=df_baseline['entry_age'])
        n_total, n_deaths = len(df_baseline), df_baseline['event'].sum()
        
        # Get percentiles
        t25 = kmf.percentile(0.25)
        t50 = kmf.percentile(0.50)
        t75 = kmf.percentile(0.75)
        
        # Calculate minimum age for relative calculations
        min_age = df_baseline['entry_age'].min()
        
        # Calculate all steepness and median metrics using the helper function
        steepness_results = calculate_steepness_from_kmf(kmf, min_age, ages_for_pdf)
        
        baseline_stats = {
            '0.25': t25, '0.50': t50, '0.75': t75,
            'n': n_total, 'n_d': n_deaths,
            'min_age': min_age,
            'steepness_iqr_absolute': steepness_results['steepness_iqr_absolute'],
            'steepness_iqr_relative': steepness_results['steepness_iqr_relative'],
            'steepness_cv_absolute': steepness_results['steepness_cv_absolute'],
            'steepness_cv_relative': steepness_results['steepness_cv_relative'],
            't_median_absolute': t50,
            't_median_relative': t50 - min_age if t50 is not None else None,
            'pdf': steepness_results['pdf'],
            'ages': steepness_results['ages']
        }
        
        results['baseline']['with_extrinsic'] = baseline_stats
        
        # Calculate baseline without extrinsic mortality using MGG model
        print("  -> Fitting MGG model for baseline without extrinsic mortality...")
        try:
            mgg = GammaGompertz()
            mgg.fit_params(
                country='USA', 
                year=2000, 
                gender='both', 
                data_type='period', 
                haz_type='mx',
                filter_from=20, 
                filter_to=105,
                print_out=False
            )
            death_times_no_extrinsic = mgg.sample_death_times(
                n=100000, 
                min_age=min_age,
                params={'a': mgg.a, 'b': mgg.b, 'c': mgg.c, 'm': 0}
            )
            kmf_no_ext = KaplanMeierFitter()
            df_no_ext = pd.DataFrame({
                'exit_age': death_times_no_extrinsic,
                'event': 1,  # All are deaths
                'entry_age': min_age  # All start at min_age
            })
            kmf_no_ext.fit(df_no_ext['exit_age'], event_observed=df_no_ext['event'], entry=df_no_ext['entry_age'])
            t25_no_ext = kmf_no_ext.percentile(0.25)
            t50_no_ext = baseline_stats['t_median_absolute']
            t75_no_ext = kmf_no_ext.percentile(0.75)
            baseline_no_ext_stats = {
                'n': len(death_times_no_extrinsic),
                'n_d': len(death_times_no_extrinsic),
                'min_age': min_age,
                '0.25': t25_no_ext, '0.50': t50_no_ext, '0.75': t75_no_ext
            }
            steepness_results_no_ext = calculate_steepness_from_kmf(kmf_no_ext, min_age, ages_for_pdf)
            baseline_no_ext_stats.update({
                'steepness_iqr_absolute': steepness_results_no_ext['steepness_iqr_absolute'],
                'steepness_iqr_relative': steepness_results_no_ext['steepness_iqr_relative'],
                'steepness_cv_absolute': steepness_results_no_ext['steepness_cv_absolute'],
                'steepness_cv_relative': steepness_results_no_ext['steepness_cv_relative'],
                't_median_absolute': t50_no_ext,
                't_median_relative': t50_no_ext - min_age,
                'pdf': steepness_results_no_ext['pdf'],
                'ages': steepness_results_no_ext['ages']
            })
            results['baseline']['without_extrinsic'] = baseline_no_ext_stats
        except Exception as e:
            print(f"  -> Warning: Could not calculate baseline without extrinsic mortality: {e}")
            results['baseline']['without_extrinsic'] = None
        print(f"  -> Baseline with extrinsic: n={n_total}, deaths={n_deaths}")
    else:
        print("  -> No baseline data available.")

    for topic_name, config in configs.items():
        print(f"Processing topic: {topic_name}...")
        results[topic_name] = {}
        try:
            df_topic = get_topic_df(topic_name, nhanes_data_path)
        except Exception as e:
            print(f"  -> Could not load data for topic {topic_name}: {e}")
            continue
        df_grouped, group_col = _apply_grouping_strategy(df_topic, config)
        required_cols = [group_col, 'entry_age', 'exit_age', 'event']
        if config.get('topic_column') and config.get('topic_column') != group_col:
            required_cols.append(config['topic_column'])
        df_topic_clean = df_grouped.dropna(subset=required_cols)
        if df_topic_clean.empty:
            print(f"  -> No data for topic {topic_name} after cleaning.")
            continue
        # Apply filtering using centralized group names
        try:
            expected_groups = get_expected_group_names(topic_name)
            available_groups = df_topic_clean[group_col].unique()
            valid_groups = [g for g in expected_groups if g in available_groups]
            
            # Apply topic-specific filtering
            if topic_name == 'alcohol' and len(valid_groups) >= 2:
                # Only keep first and last group for alcohol
                keep_groups = [valid_groups[0], valid_groups[-1]]
            elif topic_name == 'education_level':
                # Exclude 'high school' group for education_level
                keep_groups = [g for g in valid_groups if g != 'high school']
            else:
                # Keep all valid groups
                keep_groups = valid_groups
            
            if keep_groups:
                df_topic_clean = df_topic_clean[df_topic_clean[group_col].isin(keep_groups)]
        except (ValueError, KeyError):
            # If centralized approach fails, continue with all available groups
            pass
        for group_name in sorted(df_topic_clean[group_col].unique()):
            if pd.isna(group_name): continue
            subset = df_topic_clean[df_topic_clean[group_col] == group_name]
            n_subset, n_deaths_subset = len(subset), subset['event'].sum()
            if n_subset < 10:
                print(f"  -> Skipping group '{group_name}' (size < 10)")
                continue
            print(f"  -> Group '{group_name}': n={n_subset}, deaths={n_deaths_subset}")
            kmf.fit(subset['exit_age'], event_observed=subset['event'], entry=subset['entry_age'], timeline=ages_for_pdf)
            t25 = kmf.percentile(0.25)
            t50 = kmf.percentile(0.50)
            t75 = kmf.percentile(0.75)
            min_age = subset['entry_age'].min()
            steepness_results = calculate_steepness_from_kmf(kmf, min_age, ages_for_pdf)
            # Bootstrapping for errorbars
            boot_steepness_iqr_absolute = []
            boot_steepness_iqr_relative = []
            boot_steepness_cv_absolute = []
            boot_steepness_cv_relative = []
            boot_t_median_absolute = []
            for boot_i in range(n_bootstrap):
                boot_idx = np.random.choice(subset.index, size=len(subset), replace=True)
                boot_subset = subset.loc[boot_idx]
                try:
                    kmf_boot = KaplanMeierFitter()
                    kmf_boot.fit(
                        boot_subset['exit_age'],
                        event_observed=boot_subset['event'],
                        entry=boot_subset['entry_age'],
                        timeline=ages_for_pdf
                    )
                    boot_steep = calculate_steepness_from_kmf(kmf_boot, boot_subset['entry_age'].min(), ages_for_pdf)
                    # Debug print if any metric is None
                    for key in ['steepness_iqr_absolute', 'steepness_iqr_relative', 'steepness_cv_absolute', 'steepness_cv_relative']:
                        if boot_steep.get(key) is None:
                            print(f"    Bootstrap {boot_i+1}/{n_bootstrap} WARNING: {key} is None. boot_steep={boot_steep}")
                    boot_steepness_iqr_absolute.append(boot_steep['steepness_iqr_absolute'])
                    boot_steepness_iqr_relative.append(boot_steep['steepness_iqr_relative'])
                    boot_steepness_cv_absolute.append(boot_steep['steepness_cv_absolute'])
                    boot_steepness_cv_relative.append(boot_steep['steepness_cv_relative'])
                    # Median lifespan (t_median_absolute)
                    t50_boot = kmf_boot.percentile(0.50)
                    boot_t_median_absolute.append(t50_boot)
                    if print_bootstrap:
                        print(f"    Bootstrap {boot_i+1}/{n_bootstrap} done.")
                except Exception as e:
                    print(f"    Bootstrap {boot_i+1}/{n_bootstrap} failed: {e}")
                    if print_bootstrap:
                        import traceback
                        traceback.print_exc()
                    continue
            group_stats = {
                '0.25': t25, '0.50': t50, '0.75': t75,
                'n': n_subset, 'n_d': n_deaths_subset,
                'min_age': min_age,
                'steepness_iqr_absolute': steepness_results['steepness_iqr_absolute'],
                'steepness_iqr_relative': steepness_results['steepness_iqr_relative'],
                'steepness_cv_absolute': steepness_results['steepness_cv_absolute'],
                'steepness_cv_relative': steepness_results['steepness_cv_relative'],
                't_median_absolute': t50,
                't_median_relative': t50 - min_age if t50 is not None else None,
                'pdf': steepness_results['pdf'],
                'ages': steepness_results['ages'],
                # Errorbars:
                'steepness_iqr_absolute_err': safe_nanstd(boot_steepness_iqr_absolute),
                'steepness_iqr_relative_err': safe_nanstd(boot_steepness_iqr_relative),
                'steepness_cv_absolute_err': safe_nanstd(boot_steepness_cv_absolute),
                'steepness_cv_relative_err': safe_nanstd(boot_steepness_cv_relative),
                't_median_absolute_err': safe_nanstd(boot_t_median_absolute),
            }
            results[topic_name][group_name] = group_stats
    print("\nFinished calculating all survival statistics.")
    return results

# Usage example:
# interventions_dict = calculate_survival_stats(nhanes_data_path)
# interventions_dict = calculate_survival_stats(nhanes_data_path)

# Load precomputed exposure group results from pickle (now as a function, not on import)
def load_interventions_dict(pickle_path=None):
    if pickle_path is None:
        pickle_path = os.path.join(results_path, 'exposure_groups_results.pkl')
    try:
        with open(pickle_path, 'rb') as f:
            interventions_dict = pickle.load(f)
        print(f'Loaded exposure group results from {pickle_path}')
        return interventions_dict
    except FileNotFoundError:
        raise RuntimeError(f'{pickle_path} not found. Please run create_exposure_groups.py to generate it.')


def _build_sr_underlay_curves(
    results,
    from_t=20,
    steepness_metric='steepness_iqr_absolute',
    longevity_metric='t_median_absolute',
    ignore_kappa=True,
    include_h_ext=True,
):
    """Extract normalized SR steepness-longevity curves from a stored SR results dict."""
    baseline_metrics = results.get('baseline', {}).get(from_t)
    if baseline_metrics is None:
        raise KeyError(f"Missing baseline for from_t={from_t}")

    baseline_steep = baseline_metrics.get(steepness_metric)
    baseline_longev = baseline_metrics.get(longevity_metric)
    if baseline_steep is None or baseline_longev is None:
        raise KeyError(
            f"Baseline metrics missing for {steepness_metric}/{longevity_metric} at from_t={from_t}"
        )
    if baseline_steep == 0 or baseline_longev == 0:
        raise ValueError("Baseline metric is zero, cannot normalize SR underlay curves.")

    curves = {}
    params_to_use = [p for p in DEFAULT_SR_PARAM_ORDER if not (ignore_kappa and p == 'kappa')]

    for param in params_to_use:
        param_results = results.get(param, {})
        points = []
        for factor in sorted(param_results):
            metrics_by_age = param_results[factor]
            if from_t not in metrics_by_age:
                continue
            metrics = metrics_by_age[from_t]
            steep = metrics.get(steepness_metric)
            longev = metrics.get(longevity_metric)
            if steep is None or longev is None:
                continue
            if not (np.isfinite(steep) and np.isfinite(longev)):
                continue
            points.append((float(factor), float(longev / baseline_longev), float(steep / baseline_steep)))
        if points:
            curves[param] = points

    if include_h_ext and 'h_ext' in results:
        h_points = []
        for h_val in sorted(results['h_ext']):
            metrics_by_age = results['h_ext'][h_val]
            if from_t not in metrics_by_age:
                continue
            metrics = metrics_by_age[from_t]
            steep = metrics.get(steepness_metric)
            longev = metrics.get(longevity_metric)
            if steep is None or longev is None:
                continue
            if not (np.isfinite(steep) and np.isfinite(longev)):
                continue
            h_points.append((float(h_val), float(longev / baseline_longev), float(steep / baseline_steep)))
        if h_points:
            curves['h_ext'] = h_points

    return curves


def _load_or_create_sr_underlay_cache(
    source_pkl=DEFAULT_SR_USA_2019_PKL,
    cache_pkl=DEFAULT_SR_USA_2019_UNDERLAY_CACHE,
    from_t=20,
    steepness_metric='steepness_iqr_absolute',
    longevity_metric='t_median_absolute',
    ignore_kappa=True,
    include_h_ext=True,
    force_rebuild=False,
):
    """Load cached SR USA-2019 underlay curves, rebuilding cache when needed."""
    expected_metadata = {
        'source_pkl': source_pkl,
        'from_t': from_t,
        'steepness_metric': steepness_metric,
        'longevity_metric': longevity_metric,
        'ignore_kappa': ignore_kappa,
        'include_h_ext': include_h_ext,
    }

    if not force_rebuild and os.path.exists(cache_pkl):
        try:
            with open(cache_pkl, 'rb') as f:
                cached_payload = pickle.load(f)
            cached_metadata = cached_payload.get('metadata', {})
            cached_curves = cached_payload.get('curves', {})
            if cached_metadata == expected_metadata and cached_curves:
                return cached_payload
        except Exception:
            pass

    with open(source_pkl, 'rb') as f:
        sr_results = pickle.load(f)

    curves = _build_sr_underlay_curves(
        sr_results,
        from_t=from_t,
        steepness_metric=steepness_metric,
        longevity_metric=longevity_metric,
        ignore_kappa=ignore_kappa,
        include_h_ext=include_h_ext,
    )

    payload = {'metadata': expected_metadata, 'curves': curves}
    with open(cache_pkl, 'wb') as f:
        pickle.dump(payload, f)
    return payload


def save_sr_usa_2019_underlay_curves(
    source_pkl=DEFAULT_SR_USA_2019_PKL,
    cache_pkl=DEFAULT_SR_USA_2019_UNDERLAY_CACHE,
    from_t=20,
    steepness_metric='steepness_iqr_absolute',
    longevity_metric='t_median_absolute',
    ignore_kappa=True,
    include_h_ext=True,
):
    """Force-rebuild and save cached SR USA-2019 steepness-longevity underlay curves."""
    return _load_or_create_sr_underlay_cache(
        source_pkl=source_pkl,
        cache_pkl=cache_pkl,
        from_t=from_t,
        steepness_metric=steepness_metric,
        longevity_metric=longevity_metric,
        ignore_kappa=ignore_kappa,
        include_h_ext=include_h_ext,
        force_rebuild=True,
    )


def _plot_sr_usa_2019_underlay(
    ax,
    source_pkl=DEFAULT_SR_USA_2019_PKL,
    cache_pkl=DEFAULT_SR_USA_2019_UNDERLAY_CACHE,
    from_t=20,
    steepness_metric='steepness_iqr_absolute',
    longevity_metric='t_median_absolute',
    ignore_kappa=True,
    include_h_ext=True,
    alpha=0.35,
    linewidth=2.5,
    zorder=1,
):
    """Plot cached SR USA-2019 curves as a background underlay."""
    try:
        payload = _load_or_create_sr_underlay_cache(
            source_pkl=source_pkl,
            cache_pkl=cache_pkl,
            from_t=from_t,
            steepness_metric=steepness_metric,
            longevity_metric=longevity_metric,
            ignore_kappa=ignore_kappa,
            include_h_ext=include_h_ext,
            force_rebuild=False,
        )
    except Exception as exc:
        print(f"Warning: could not load SR underlay curves ({exc}).")
        return []

    curves = payload.get('curves', {})
    artists = []

    params_to_use = [p for p in DEFAULT_SR_PARAM_ORDER if not (ignore_kappa and p == 'kappa')]
    for param in params_to_use:
        points = curves.get(param, [])
        if not points:
            continue
        xs = [point[1] for point in points]
        ys = [point[2] for point in points]
        color = sr_utils.param_colors.get(param, 'gray')
        line = ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)[0]
        artists.append(line)

    if include_h_ext and curves.get('h_ext'):
        h_points = curves['h_ext']
        xs = [point[1] for point in h_points]
        ys = [point[2] for point in h_points]
        line = ax.plot(xs, ys, color='red', linewidth=linewidth, alpha=alpha, zorder=zorder)[0]
        artists.append(line)

    return artists


# =============================================================================
# Helper Functions
# =============================================================================

def load_kmf_for_group(topic_name, group_name, nhanes_data_path=None, timeline=None):
    """
    Load a Kaplan-Meier fitter for a specific topic and group.
    
    Args:
        topic_name (str): Name of the topic (must be in TOPIC_CONFIGS)
        group_name (str): Name of the group within the topic
        nhanes_data_path (str, optional): Path to NHANES data. If None, uses default path.
        timeline (array, optional): Age timeline for fitting. If None, uses default timeline.
    
    Returns:
        tuple: (kmf, group_data) where kmf is the fitted KaplanMeierFitter and group_data is the DataFrame for the group
    """
    if topic_name not in TOPIC_CONFIGS:
        raise ValueError(f"Topic '{topic_name}' not found in TOPIC_CONFIGS. Available topics: {list(TOPIC_CONFIGS.keys())}")
    
    if nhanes_data_path is None:
        nhanes_data_path = globals()["nhanes_data_path"]
    
    if timeline is None:
        timeline = np.linspace(0, 120, 1000)
    
    kmf = KaplanMeierFitter()
    config = TOPIC_CONFIGS[topic_name]
    
    # Get topic data and apply grouping strategy
    df_topic = get_topic_df(topic_name, nhanes_data_path)
    df_grouped, group_col = _apply_grouping_strategy(df_topic, config)
    
    # Clean data
    required_cols = [group_col, 'entry_age', 'exit_age', 'event']
    df_clean = df_grouped.dropna(subset=required_cols)
    
    if df_clean.empty:
        raise ValueError(f"No data available for topic '{topic_name}' after cleaning")
    
    # Get the specific group data
    group_data = df_clean[df_clean[group_col] == group_name]
    
    if group_data.empty:
        raise ValueError(f"Group '{group_name}' not found in topic '{topic_name}'. Available groups: {sorted(df_clean[group_col].unique())}")
    
    # Fit the Kaplan-Meier model
    kmf.fit(
        durations=group_data['exit_age'], 
        event_observed=group_data['event'],
        entry=group_data['entry_age'], 
        timeline=timeline
    )
    
    return kmf, group_data


def list_available_topics(interventions_dict):
    """Prints all available topics for analysis."""
    topics = [key for key in interventions_dict.keys() if key != 'baseline']
    print("Available intervention topics:")
    for topic in sorted(topics):
        print(f"- {topic}")
    
    # Also show available baseline types
    if 'baseline' in interventions_dict:
        print("\nAvailable baseline types:")
        for baseline_type in interventions_dict['baseline'].keys():
            print(f"- {baseline_type}")

def list_available_groups(topic_name, nhanes_data_path=None):
    """
    Print all available groups for a given topic.
    
    Args:
        topic_name (str): Name of the topic (must be in TOPIC_CONFIGS)
        nhanes_data_path (str, optional): Path to NHANES data. If None, uses default path.
    """
    if topic_name not in TOPIC_CONFIGS:
        print(f"Topic '{topic_name}' not found in TOPIC_CONFIGS.")
        print(f"Available topics: {list(TOPIC_CONFIGS.keys())}")
        return
    
    if nhanes_data_path is None:
        nhanes_data_path = globals()["nhanes_data_path"]
    
    try:
        config = TOPIC_CONFIGS[topic_name]
        
        # Get topic data and apply grouping strategy
        df_topic = get_topic_df(topic_name, nhanes_data_path)
        df_grouped, group_col = _apply_grouping_strategy(df_topic, config)
        
        # Clean data
        required_cols = [group_col, 'entry_age', 'exit_age', 'event']
        df_clean = df_grouped.dropna(subset=required_cols)
        
        if df_clean.empty:
            print(f"No data available for topic '{topic_name}' after cleaning.")
            return
        
        # Get unique groups and their counts
        group_counts = df_clean[group_col].value_counts().sort_index()
        
        print(f"Available groups for topic '{topic_name}':")
        print("-" * 50)
        for group_name, count in group_counts.items():
            if pd.isna(group_name):
                continue
            print(f"- {group_name}: {count} participants")
        
        print(f"\nTotal groups: {len(group_counts)}")
        print(f"Total participants: {len(df_clean)}")
        
    except Exception as e:
        print(f"Error loading groups for topic '{topic_name}': {e}")

def get_metric_from_group(group_stats, metric_name):
    """
    Helper function to safely extract a metric from group statistics.
    
    Args:
        group_stats (dict): Dictionary containing group statistics
        metric_name (str): Name of the metric to extract
        
    Returns:
        float or None: The metric value if available, None otherwise
    """
    return group_stats.get(metric_name)

def get_available_metrics(interventions_dict, topic=None, group=None):
    """
    Get list of available metrics in the interventions dictionary.
    
    Args:
        interventions_dict (dict): The interventions dictionary
        topic (str, optional): Specific topic to check
        group (str, optional): Specific group to check
        
    Returns:
        list: List of available metric names
    """
    if topic is None:
        # Check baseline for available metrics
        if 'baseline' in interventions_dict:
            baseline_types = list(interventions_dict['baseline'].keys())
            if baseline_types:
                first_baseline = interventions_dict['baseline'][baseline_types[0]]
                if first_baseline:
                    return list(first_baseline.keys())
    else:
        if topic in interventions_dict:
            if group is None:
                # Get first group to check metrics
                groups = list(interventions_dict[topic].keys())
                if groups:
                    first_group = interventions_dict[topic][groups[0]]
                    return list(first_group.keys())
            else:
                if group in interventions_dict[topic]:
                    return list(interventions_dict[topic][group].keys())
    
    return []

def print_metric_summary(interventions_dict, topic, metric_name):
    """
    Print a summary of a specific metric across all groups in a topic.
    
    Args:
        interventions_dict (dict): The interventions dictionary
        topic (str): Topic name
        metric_name (str): Metric name to summarize
    """
    if topic not in interventions_dict:
        print(f"Topic '{topic}' not found")
        return
    
    print(f"\nMetric summary for {topic} - {metric_name}:")
    print("-" * 50)
    
    for group_name, group_stats in interventions_dict[topic].items():
        value = get_metric_from_group(group_stats, metric_name)
        if value is not None:
            print(f"{group_name}: {value:.4f}")
        else:
            print(f"{group_name}: Not available")

# -------- helper: lighten colour towards white ---------------
def _shade_series(base_hex, n):
    """Return n progressively lighter shades of base_hex (dark → light)."""
    base = np.array(mcolors.to_rgb(base_hex))
    t    = np.linspace(0.15, 0.6, n)      # 0 = pure colour, 1 = white
    return [mcolors.to_hex((1-s)*base + s) for s in t]

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_survival_by_topic(topic_name, nhanes_data_path=None, timeline=np.linspace(0, 120, 1000), ax=None, show_baseline=True, with_extrinsic=True):
    """
    Plots Kaplan-Meier survival curves for a given topic using TOPIC_CONFIGS.
    
    Args:
        topic_name (str): Name of the topic (must be in TOPIC_CONFIGS)
        nhanes_data_path (str, optional): Path to NHANES data. If None, uses default path.
        timeline (array, optional): Age timeline for plotting. Defaults to 0-120 years.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
        show_baseline (bool, optional): Whether to show baseline survival curve. Defaults to True.
        with_extrinsic (bool, optional): Whether to include extrinsic mortality in baseline. Defaults to True.
    
    Returns:
        matplotlib.axes.Axes: The axes object with the plot
    """
    if topic_name not in TOPIC_CONFIGS:
        raise ValueError(f"Topic '{topic_name}' not found in TOPIC_CONFIGS. Available topics: {list(TOPIC_CONFIGS.keys())}")
    
    if nhanes_data_path is None:
        nhanes_data_path = globals()["nhanes_data_path"]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    config = TOPIC_CONFIGS[topic_name]
    
    # Get topic data and apply grouping strategy to find available groups
    df_topic = get_topic_df(topic_name, nhanes_data_path)
    df_grouped, group_col = _apply_grouping_strategy(df_topic, config)
    
    # Clean data
    required_cols = [group_col, 'entry_age', 'exit_age', 'event']
    df_clean = df_grouped.dropna(subset=required_cols)
    
    if df_clean.empty:
        raise ValueError(f"No data available for topic '{topic_name}' after cleaning")
    
    print(f"Value counts for {topic_name}:\n{df_clean[group_col].value_counts()}")
    
    # Plot baseline if requested
    if show_baseline:
        plot_baseline_km(ax)
    
    # Plot each group using the helper function
    for group_name in sorted(df_clean[group_col].unique()):
        if pd.isna(group_name):
            continue
            
        try:
            kmf, group_data = load_kmf_for_group(topic_name, group_name, nhanes_data_path, timeline)
            label = f'{group_name} (n={len(group_data)})'
            kmf.plot_survival_function(ax=ax, label=label)
        except Exception as e:
            print(f"Warning: Could not plot group '{group_name}': {e}")
            continue

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(config['title'])
    ax.legend(title=group_col.replace('_', ' ').title())
    plt.tight_layout()
    
    return ax


def plot_intervention_steepness_longevity(
        interventions_dict=None,
        topics=('number_of_friends',),
        ax=None,
        steepness_metric='steepness_iqr_absolute',
        longevity_metric='t_median_absolute',
        baseline_type='with_extrinsic',
        label_points=True,
        adjust_text_kwargs=None,
        marker_size=None,
        legend_fontsize=None,
        xlim=None,
        ylim=None,
        error_bars=True,
        use_colors=False,
        sr_underlay=True,
        sr_underlay_from_t=20,
        sr_underlay_source_pkl=DEFAULT_SR_USA_2019_PKL,
        sr_underlay_cache_pkl=DEFAULT_SR_USA_2019_UNDERLAY_CACHE,
        sr_underlay_ignore_kappa=True,
        sr_underlay_include_h_ext=True,
        sr_underlay_alpha=0.35,
        sr_underlay_linewidth=2.5,
        zorder=100,
        **kwargs):
    """
    Scatter plot: x = median lifespan / baseline, y = steepness / baseline.
    ▸ colour family = topic               (single legend entry per topic)
    ▸ shade within family = intensity     (no legend; label text instead)
    ▸ text labels positioned with adjustText.
    Now includes errorbars for steepness (y-axis) and median lifespan (x-axis) if available and error_bars is True.
    Uses group labels and legend labels from TOPIC_CONFIGS.
    If sr_underlay=True, underlays the cached USA-2019 SR steepness-longevity curves.
    """
    if interventions_dict is None:
        interventions_dict = load_interventions_dict()

    point_size   = marker_size if marker_size is not None else kwargs.get('s', 1000)
    marker_style = kwargs.get('marker', 's')

    # Set Arial font for all text elements
    plt.rcParams['font.family'] = 'Arial'

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # ---------- baseline ----------------------------------------
    base_stats = interventions_dict.get('baseline', {}).get(baseline_type)
    if not base_stats:
        raise ValueError(f"baseline → {baseline_type} → metrics not found")
    longev_base = base_stats.get(longevity_metric)
    steep_base = base_stats.get(steepness_metric)
    if longev_base is None or steep_base is None:
        raise ValueError(f"Baseline {longev_base} or {steep_base} not available")

    if sr_underlay:
        _plot_sr_usa_2019_underlay(
            ax=ax,
            source_pkl=sr_underlay_source_pkl,
            cache_pkl=sr_underlay_cache_pkl,
            from_t=sr_underlay_from_t,
            steepness_metric=steepness_metric,
            longevity_metric=longevity_metric,
            ignore_kappa=sr_underlay_ignore_kappa,
            include_h_ext=sr_underlay_include_h_ext,
            alpha=sr_underlay_alpha,
            linewidth=sr_underlay_linewidth,
            zorder=max(1, zorder - 50),
        )

    # ------------------------------------------------------------------
    #  Prepare containers for scatter points and their labels
    # ------------------------------------------------------------------
    all_xs, all_ys = [], []
    all_texts = []

    for topic in topics if isinstance(topics, (list, tuple)) else [topics]:
        groups = interventions_dict.get(topic)
        if not groups:
            print(f"topic '{topic}' missing → skipped")
            continue
        
        # Get expected group names from centralized function
        try:
            expected_groups = get_expected_group_names(topic)
            # Only keep groups that exist in both expected and actual data
            intensities = [g for g in expected_groups if g in groups]
        except (ValueError, KeyError):
            # Fallback to sorted group keys if centralized approach fails
            intensities = sorted(groups.keys())
        
        # Apply filtering logic for specific topics
        if topic == 'alcohol':
            # Only keep first and last group for alcohol
            if len(intensities) >= 2:
                intensities = [intensities[0], intensities[-1]]
        elif topic == 'education_level':
            # Exclude 'high school' group for education_level
            intensities = [i for i in intensities if i != 'high school']
        
        xs, ys, xerrs, yerrs = [], [], [], []
        cs = []
        
        if use_colors:
            # Get config for legend label
            config = TOPIC_CONFIGS.get(topic, {})
            # Use legend label from config if available
            legend_label = config.get('legend_label', topic.replace('_', ' ').title())
            reverse_shade = topic in ['diet', 'income', 'sleep_frailty']
            # Special alpha reversal for specific topics (opposite of reverse_shade for these)
            reverse_alpha = topic in ['alcohol', 'sleep_duration']
            # Use same base color for all intensities of this topic
            base_color = TOPIC_COLOURS[topic]
            
            # Determine per-group alpha values (progressively increasing)
            if len(intensities) > 1:
                alphas = np.linspace(0.4, 1.0, len(intensities))
                # Reverse alphas for reverse_shade topics OR reverse_alpha topics
                if reverse_shade or reverse_alpha:
                    alphas = alphas[::-1]
            else:
                alphas = [1.0]

        for intensity in intensities:
            g = groups[intensity]
            longev = g.get(longevity_metric)
            steep = g.get(steepness_metric)
            
            # Errorbars for steepness and median lifespan
            steep_err = g.get(steepness_metric + '_err', None)
            longev_err = g.get('t_median_absolute_err', None)
            if longev is None or steep is None:
                continue
            if not (np.isfinite(longev) and np.isfinite(steep)):
                continue
            x_val = longev / longev_base
            y_val = steep / steep_base
            
            xs.append(x_val)
            ys.append(y_val)

            if use_colors:
                # Apply the alpha that corresponds to this intensity's position
                alpha_val = alphas[intensities.index(intensity)] if intensity in intensities else 1.0
                cs.append(mcolors.to_rgba(base_color, alpha_val))

            # Calculate errorbars relative to baseline
            if error_bars:
                # For x (median lifespan): propagate error if available
                if longev_err is not None and np.isfinite(longev_err) and longev_base != 0:
                    xerrs.append(longev_err / longev_base)
                else:
                    xerrs.append(0)
                # For y (steepness): propagate error if available
                if steep_err is not None and np.isfinite(steep_err) and steep_base != 0:
                    yerrs.append(steep_err / steep_base)
                else:
                    yerrs.append(0)
            else:
                xerrs.append(0)
                yerrs.append(0)
        
        if not xs:
            continue
        
        jitter = 0.003
        xs_jittered = [x + np.random.uniform(-jitter, jitter) for x in xs]
        ys_jittered = [y + np.random.uniform(-jitter, jitter) for y in ys]
        xerrs = np.array(xerrs)
        yerrs = np.array(yerrs)

        # Plot errorbars if any are nonzero and error_bars is True
        if error_bars and (np.any(xerrs > 0) or np.any(yerrs > 0)):
            ecolor = 'gray' if use_colors else 'black'
            elinewidth = 2 if use_colors else 1
            capsize = 5 if use_colors else 3
            ax.errorbar(xs_jittered, ys_jittered, xerr=xerrs, yerr=yerrs, fmt='none', ecolor=ecolor, elinewidth=elinewidth, capsize=capsize, zorder=zorder)

        if use_colors:
            marker_shape = TOPIC_MARKERS.get(topic, marker_style)
            ax.scatter(xs_jittered, ys_jittered, s=point_size, c=cs, marker=marker_shape, edgecolors='w', zorder=zorder+1)
        else:
            marker_shape = BW_TOPIC_MARKERS.get(topic, marker_style)
            if topic == 'sleep_frailty':
                ax.scatter(xs_jittered, ys_jittered, s=point_size, marker=marker_shape, facecolors='white', edgecolors='black', linewidths=1, zorder=zorder+1)
            else:
                ax.scatter(xs_jittered, ys_jittered, s=point_size, marker=marker_shape, c='black', zorder=zorder+1)

        for xj, yj, intensity in zip(xs_jittered, ys_jittered, intensities):
            txt = ax.text(xj, yj, intensity,
                          fontsize=14, fontweight='normal', ha='center', va='center',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.5),
                          zorder=1000)
            all_texts.append(txt)
        all_xs.extend(xs_jittered)
        all_ys.extend(ys_jittered)

    # ------------------------------------------------------------------
    #  Label adjustment – apply adjust_text for ALL metrics (IQR & CV alike)
    # ------------------------------------------------------------------
    if all_texts and label_points:
        # Set tight axis limits from data BEFORE adjust_text so labels stay
        # near the visible area (SR underlay curves can extend far and blow up
        # the auto-scaled range, causing adjust_text to place labels at extreme
        # positions that later trigger huge bbox_inches='tight' allocations).
        if all_xs and all_ys:
            pad_x = 0.05 * (max(all_xs) - min(all_xs)) if max(all_xs) != min(all_xs) else 0.05
            pad_y = 0.15 * (max(all_ys) - min(all_ys)) if max(all_ys) != min(all_ys) else 0.15
            ax.set_xlim(xlim or (min(all_xs) - pad_x, max(all_xs) + pad_x))
            ax.set_ylim(ylim or (min(all_ys) - pad_y, max(all_ys) + pad_y))
        else:
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
        adjust_kwargs = dict(
            ax=ax,
            expand=(1.2, 1.5),
            ensure_inside_axes=True,
            arrowprops=dict(
                arrowstyle="->",
                color='gray',
                lw=0.5,
                shrinkA=5,
                shrinkB=5
            )
        )
        if adjust_text_kwargs:
            adjust_kwargs.update(adjust_text_kwargs)
        adjust_text(all_texts, **adjust_kwargs)

    # --------- cosmetics ---------------------------------------
    longevity_label = 'Median' if 'median' in longevity_metric else 'Maximum'
    steepness_label = 'IQR' if 'iqr' in steepness_metric else 'CV'
    relative_label = ' (Relative)' if 'relative' in longevity_metric else ' (Absolute)'
    ax.set_xlabel(f"Median lifespan Exposure / Control", fontsize=legend_fontsize if legend_fontsize is not None else 18)
    ax.set_ylabel(f"Steepness Exposure / Control", fontsize=legend_fontsize if legend_fontsize is not None else 18)
    # Remove grid and add faint reference lines at x=1 and y=1
    ax.axvline(1, color='grey', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax.axhline(1, color='grey', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

    topics_list = topics if isinstance(topics, (list, tuple)) else [topics]
    legend_handles = []
    legend_labels = []
    for topic in topics_list:
        config = TOPIC_CONFIGS.get(topic, {})
        legend_label = config.get('legend_label', topic.replace('_', ' ').title())
        # Scale marker size with legend font size
        legend_marker_size = (legend_fontsize if legend_fontsize is not None else 18) * 0.6
        
        if use_colors:
            color = TOPIC_COLOURS.get(topic, 'gray')
            marker = TOPIC_MARKERS.get(topic, 'o')
            if marker in ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'h', '8', '<']:
                handle = mlines.Line2D([], [], color=color, marker=marker, linestyle='None', markersize=legend_marker_size, label=legend_label, markerfacecolor=color)
            else:
                handle = mpatches.Patch(color=color, label=legend_label)
        else:
            marker = BW_TOPIC_MARKERS.get(topic, 'o')
            facecolor = 'white' if topic == 'sleep_frailty' else 'black'
            edgecolor = 'black'
            handle = mlines.Line2D([], [], color='black', marker=marker, linestyle='None', 
                                   markersize=legend_marker_size, label=legend_label, 
                                   markerfacecolor=facecolor, markeredgecolor=edgecolor)
        legend_handles.append(handle)
        legend_labels.append(legend_label)
        
    # Set legend with custom title and larger title font
    legend = ax.legend(handles=legend_handles, title="Exposures", frameon=True, fontsize=legend_fontsize if legend_fontsize is not None else 18, title_fontsize=(legend_fontsize + 2 if legend_fontsize is not None else 16))
    # Make the legend title bold for emphasis
    if legend and legend.get_title():
        legend.get_title().set_fontweight('bold')

    baseline_label = "with extrinsic" if baseline_type == 'with_extrinsic' else "without extrinsic"
    ax.set_title(f"Intervention landscape ({steepness_metric}, {longevity_metric}, {baseline_label})")
    return ax, legend 


# ------------  global palette  (topic -> base colour)  -----------
TOPIC_COLOURS = {
    'diet'              : "#1f77b4",  # blue
    'income'            : "#9467bd",  # purple
    'alcohol'           : "#ff7f0e",  # orange
    'physical_activity' : "#2ca02c",  # green
    'sleep_duration'    : "#ffd700",  # yellow
    'sleep_frailty'     : "#8c564b",  # brown
    'work_regularity'   : "#d62728",  # red
    'number_of_friends' : "#7f7f7f",  # grey
    'church_frequency'  : "#17becf",  # cyan
    'education_level'   : "#e377c2",  # pink
}
# ------------  global palette  (topic -> marker) for B&W plots -----------
BW_TOPIC_MARKERS = {
    'diet'              : '$D$',
    'income'            : 'o',
    'alcohol'           : 'D',
    'physical_activity' : '^',
    'sleep_duration'    : 'v',
    'sleep_frailty'     : 'v',
    'education_level'   : '<',
    'church_frequency'  : '$\\dagger$',
    'number_of_friends' : '$f$',
    'work_regularity'   : 'X',
}
# ------------  global palette  (topic -> marker)  -----------
TOPIC_MARKERS = {
    'diet'              : 's',  # square
    'income'            : 'o',  # circle
    'alcohol'           : 'D',  # diamond
    'physical_activity' : '^',  # triangle up
    'sleep_duration'    : 'v',  # triangle down
    'sleep_frailty'     : 'P',  # plus (filled)
    'work_regularity'   : 'X',  # x (filled)
    'number_of_friends' : 'h',  # hexagon
    'church_frequency'  : '8',  # octagon
    'education_level'   : '<',  # triangle left
}



def create_full_df(nhanes_data_path=None):
    """
    Combine all topic DataFrames into one, with one row per SEQN (person).
    Columns: SEQN, entry_age, exit_age, mortality_status, and one clear column per topic.
    Missing data is NaN. Uses outer join on SEQN.
    """
    if nhanes_data_path is None:
        nhanes_data_path = globals()["nhanes_data_path"]
    topic_column_map = {
        'diet': ('diet', 'diet_quality'),
        'income': ('INDFMMPI', 'income_ratio'),
        'alcohol': ('drinks_per_day', 'alcohol_drinks_per_day'),
        'physical_activity': ('physical_activity_index', 'physical_activity_index'),
        'sleep_duration': ('sleep_hours', 'sleep_duration_hours'),
        'sleep_frailty': ('sleep_frailty', 'sleep_frailty_index'),
        'work_regularity': ('work_schedule', 'work_schedule'),
        'number_of_friends': ('number_of_friends', 'number_of_friends'),
        'church_frequency': ('church_frequency', 'church_frequency'),
        'education_level': ('education_level', 'education_level'),
    }
    # Start with the core DataFrame for SEQN, entry_age, exit_age, event
    core = load_core(nhanes_data_path)[['SEQN', 'entry_age', 'exit_age', 'event']]
    # Map event to binary dead column
    core = core.copy()
    core['dead'] = core['event'].map({1: 1, 0: 0}).astype('Int64')
    # Only keep SEQN, entry_age, exit_age, dead
    full_df = core[['SEQN', 'entry_age', 'exit_age', 'dead']]
    # For each topic, merge in the topic column
    for topic, (col, new_col) in topic_column_map.items():
        try:
            df_topic = get_topic_df(topic, nhanes_data_path)
            # Only keep SEQN and the topic column
            if col in df_topic.columns:
                df_topic = df_topic[['SEQN', col]].rename(columns={col: new_col})
            else:
                # For work_regularity, col is 'work_schedule', but df may also have 'work_hours'
                # Just keep SEQN and all columns containing the col string
                topic_cols = [c for c in df_topic.columns if col in c]
                if topic_cols:
                    df_topic = df_topic[['SEQN'] + topic_cols]
                    df_topic = df_topic.rename(columns={topic_cols[0]: new_col})
                else:
                    continue
            # Outer merge to keep all SEQNs
            full_df = pd.merge(full_df, df_topic, on='SEQN', how='outer')
        except Exception as e:
            print(f"Skipping topic {topic} due to error: {e}")
            continue
    # Ensure integer columns are correct dtype after merge
    int_cols = ['number_of_friends', 'diet_quality', 'alcohol_drinks_per_day', 'sleep_duration_hours', 'need_for_support']
    for col in int_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].round().astype('Int64')
    return full_df


def build_exposure_score_df(nhanes_data_path=None, include_work_regularity=False):
    """
    Build complete-case cohort across exposure topics and compute additive exposure score.

    Scoring rule (1 point each):
    - diet: Good
    - alcohol: 0-1 drink/day
    - physical_activity: Some Activity
    - sleep_duration: 5-<7 hours OR 7-<9 hours
    - sleep_frailty: Q1 (lowest)
    - number_of_friends: 1+ friends
    - church_frequency: weekly
    - education_level: some college
    - income: Q3 OR Q4 (Highest)

    Returns:
        pd.DataFrame: one row per SEQN in complete-case cohort with grouped topic labels
                     and columns:
                     - exposure_score
                     - points_<topic> (0/1 per topic)
    """
    if nhanes_data_path is None:
        nhanes_data_path = globals()["nhanes_data_path"]

    topics = list(TOPIC_CONFIGS.keys())
    if not include_work_regularity:
        topics = [t for t in topics if t != 'work_regularity']

    per_topic_grouped = []
    for topic in topics:
        config = TOPIC_CONFIGS[topic]
        df_topic = get_topic_df(topic, nhanes_data_path)
        df_topic = df_topic.dropna(subset=[c for c in ['entry_age', 'exit_age', 'event'] if c in df_topic.columns])
        grouped_df, group_col = _apply_grouping_strategy(df_topic, config)
        grouped_df = grouped_df[['SEQN', group_col]].dropna(subset=[group_col]).drop_duplicates(subset='SEQN', keep='last').copy()
        grouped_df[topic] = grouped_df[group_col].astype(str)
        per_topic_grouped.append(grouped_df[['SEQN', topic]])

    if not per_topic_grouped:
        return pd.DataFrame(columns=['SEQN', 'exposure_score'])

    score_df = per_topic_grouped[0]
    for next_df in per_topic_grouped[1:]:
        score_df = score_df.merge(next_df, on='SEQN', how='inner')

    # Point mapping by grouped labels
    point_rules = {
        'diet': lambda x: x == 'Good',
        'income': lambda x: x in ['Q3', 'Q4 (Highest)'],
        'alcohol': lambda x: x == '0-1 drink/day',
        'physical_activity': lambda x: x == 'Some Activity',
        'sleep_duration': lambda x: x in ['5-<7 hours', '7-<9 hours'],
        'sleep_frailty': lambda x: x == 'Q1 (lowest)',
        'number_of_friends': lambda x: x == '1+ friends',
        'church_frequency': lambda x: x == 'weekly',
        'education_level': lambda x: x == 'some college',
        # Work regularity intentionally has no point rule in requested scheme
    }

    point_cols = []
    for topic in topics:
        if topic in point_rules:
            pcol = f'points_{topic}'
            score_df[pcol] = score_df[topic].apply(lambda v: 1 if point_rules[topic](v) else 0).astype(int)
            point_cols.append(pcol)

    score_df['exposure_score'] = score_df[point_cols].sum(axis=1).astype(int) if point_cols else 0
    return score_df


def summarize_exposure_score(nhanes_data_path=None, include_work_regularity=False):
    """
    Return summary table for additive exposure score distribution.

    Returns:
        pd.DataFrame with columns: exposure_score, n, pct
    """
    score_df = build_exposure_score_df(
        nhanes_data_path=nhanes_data_path,
        include_work_regularity=include_work_regularity,
    )
    if score_df.empty:
        return pd.DataFrame(columns=['exposure_score', 'n', 'pct'])

    summary = score_df['exposure_score'].value_counts().sort_index().rename_axis('exposure_score').reset_index(name='n')
    total_n = summary['n'].sum()
    summary['pct'] = (summary['n'] / total_n) * 100.0
    return summary
