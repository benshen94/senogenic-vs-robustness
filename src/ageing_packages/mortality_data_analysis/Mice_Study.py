import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, NelsonAalenFitter

class MiceStudy:
  def __init__(self, year='all', sex='both', site='all'):
      repo_root = Path(__file__).resolve().parents[3]
      self.data_folder = os.environ.get(
          "SENOGENIC_MICE_DATA_DIR",
          str(repo_root / "data" / "mice"),
      )
      self.year = year
      self.sex = sex
      self.site = site
      self.data = self.load_data()
      self.kmf = KaplanMeierFitter()
      self.naf = self._create_naf()
      self.fit_survival()
      self.med_time = self.calculate_median_time()
      self.inter_quartile = self.calculate_interquartile_range()
      self.steepness = self.med_time / self.inter_quartile

  def load_data(self):
      if self.year == 'all':
          return self.load_all_years_data()
      else:
          return self.load_single_year_data()

  def load_single_year_data(self):
      filename = os.path.join(self.data_folder, f'Lifespan_C{self.year}.xlsx')
      data = pd.read_excel(filename)
      data.columns = data.columns.str.lower()
      data = data[data['group'] == 'Control']
      if self.sex in ['f', 'm']:
          data = data[data['sex'] == self.sex]
      if self.site != "all":
          data = data[data['site'] == self.site]
      return data

  def load_all_years_data(self):
      all_data = []

      for file in os.listdir(self.data_folder):
          if file.endswith('.xlsx'):
              file_path = os.path.join(self.data_folder, file)
              data = pd.read_excel(file_path)
              data.columns = data.columns.str.lower()
              data = data[data['group'] == 'Control']
              if self.sex in ['f', 'm']:
                  data = data[data['sex'] == self.sex]
              if self.site != "all":
                  data = data[data['site'] == self.site]
              all_data.append(data)

      combined_data = pd.concat(all_data, ignore_index=True)
      return combined_data

  def _create_naf(self):
        event_observed = self.data['status'] == 'dead'
        naf = NelsonAalenFitter()
        naf.fit(self.data['age'], event_observed=event_observed)
        return naf

  def fit_survival(self):
      lifetimes = self.data['age']

      # Determine if the event was observed (1 if dead, 0 if censored)
      event_observed = self.data['status'] == 'dead'

      # Fit the Kaplan-Meier and Nelson-Aalen models
      self.kmf.fit(durations=lifetimes, event_observed=event_observed)
      self.naf.fit(durations=lifetimes, event_observed=event_observed)

  def calculate_median_time(self):
      return self.kmf.median_survival_time_

  def calculate_interquartile_range(self):
      q75 = self.kmf.percentile(0.75)
      q25 = self.kmf.percentile(0.25)
      return q75 - q25

  def plot_survival(self, ax=None, **kwargs):
      if ax is None:
          fig, ax = plt.subplots(figsize=(10, 6))
      self.kmf.plot_survival_function(ax=ax, **kwargs)
      ax.set_xlabel('Age (days)')
      ax.set_ylabel('Survival Probability')
      ax.set_title(f'Survival Curve for {self.year}, {self.sex.capitalize()}, {self.site.upper()}')
      if ax is None:
          plt.show()

  def plot_hazard(self, ax=None, bandwidth = 3, **kwargs):
      if ax is None:
          fig, ax = plt.subplots(figsize=(10, 6))
      self.naf.plot_hazard(bandwidth=bandwidth, ax=ax, **kwargs)
      ax.set_xlabel('Age (days)')
      ax.set_ylabel('Hazard Rate [1/days]')
      ax.set_yscale('log')
      ax.set_title(f'Hazard Rate for {self.year}, {self.sex.capitalize()}, {self.site.upper()}')
      if ax is None:
          plt.show()

  def plot_cumulative_hazard(self, ax=None, **kwargs):
      if ax is None:
          fig, ax = plt.subplots(figsize=(10, 6))
      self.naf.plot_cumulative_hazard(ax=ax, **kwargs)
      ax.set_xlabel('Age (days)')
      ax.set_ylabel('Cumulative Hazard')
      ax.set_title(f'Cumulative Hazard for {self.year}, {self.sex.capitalize()}, {self.site.upper()}')
      if ax is None:
          plt.show()
