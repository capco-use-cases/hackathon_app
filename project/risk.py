import pandas as pd, numpy as np
import pickle, datetime
import warnings
warnings.filterwarnings('ignore')



class prediction:

  def __init__(self,model_path,scaler_path):

    # self.fields = prediction.preprocessing(**fields)
    with open(model_path, 'rb') as f:
      self.risk_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
      self.scaler = pickle.load(f)
    # with open(recommender_model_path, 'rb') as f:
    #   self.recommender = pickle.load(f)


  def risk_preprocessing(self, **fields):
  # Basic preprocessing
    df = pd.DataFrame(dict(zip(fields.keys(), [[i] for i in fields.values()])))

    mappings = dict(zip(['10+ years', '8 years', '6 years', '3 years', '< 1 year',
        '4 years', '5 years', '2 years', '9 years', '1 year', '7 years'],
    [10, 8, 6, 3, 0, 4, 5, 2, 9, 1, 7]))
    df.years_in_current_job = df.years_in_current_job.map(mappings)

    # print(df)

    # Loan type in req
    risks = {'Business Loan': 31.88518231186967,
  'Personal Loan': 22.84722468208707,
  'other': 21.49664114982034,
  'shopping loan': 21.45748987854251,
  'Education Loan': 20.28985507246377,
  'Home Loan': 19.98247151621385,
  'wedding loan': 19.54022988505747,
  'Vehicle Loan': 16.34506242905789}
    qnt = pd.DataFrame(pd.Series(risks))
    qnt.columns = ['loan_type_risk_pct']
    # qnt = pd.DataFrame(dic).set_index('Unnamed: 0', drop=True)
    df = pd.merge(df, qnt, left_on='loan_type', right_index=True, how='inner')
    loan_type_mappings = dict(zip(qnt.index, [1,2]+[3]*6))
    loan_type_mappings = dict([(key,'_'.join(key.lower().split())) if value != 3 else (key,'other') for key,value in loan_type_mappings.items()])
    df.loan_type = df.loan_type.map(loan_type_mappings)

    if fields['loan_type'] not in ['Business Loan', 'Personal Loan']:
      loan_type_other = 1
    else:
      loan_type_other = 0
    df.loc[0, 'loan_type_other'] = loan_type_other
    # df = pd.get_dummies(df, columns=['loan_type'], prefix='loan_type', drop_first=True)
    # df.columns = ['_'.join(i.lower().split()) for i in df.columns]

    df.loc[:,df.select_dtypes(exclude=['float64', 'object']).columns] = df.select_dtypes(exclude=['float64', 'object']).astype('int64')


    # household income is req
    if (fields['household_income'] >= 50_000) and (fields['household_income'] < 300_000):
      middle_class = 1
    else:
      middle_class = 0
    df.loc[0, 'social_class_middle_class'] = middle_class
    # df['social_class'] = pd.cut(df.household_income, [0, 12_500, 50_000, 300_000], labels=['poor', 'aspirer', 'middle_class']).astype('object')
    # df.loc[(df.household_income>=300_000), 'social_class'] = 'rich'
    # df = pd.get_dummies(df, columns=['social_class'], drop_first=True)


    df['loan_to_income_ratio'] = df.current_loan_amount / df.household_income
    df.loc[df['loan_to_income_ratio'] >= 0.4, 'excessive_loan'] = 1
    df.loc[((df['loan_to_income_ratio'] >= 0.3) & (df['loan_to_income_ratio'] < 0.4)), 'normal_loan'] = 1
    df.loc[:,['excessive_loan','normal_loan']] = df.loc[:,['excessive_loan','normal_loan']].fillna(0)

    df['monthlydebt_to_income_ratio'] = df.monthly_debt / (df.household_income/12)
    df.loc[df['monthlydebt_to_income_ratio'] > 0.36, 'excessive_debt'] = 1
    # df = df.drop(['monthlyDebt_to_income_ratio', 'monthly_debt', 'loan_to_income_ratio', 'current_loan_amount'], axis=1)
    df.loc[:,['excessive_debt']] = df.loc[:,['excessive_debt']].fillna(0)


    df['bad_financial_condition'] = df.excessive_debt + df.excessive_loan

    df = df.select_dtypes(exclude='object')

    # print(df.columns)
    # return df
    # print(df)
    # non_binary_fields = ['years_of_credit_history','years_in_current_job',
    #       'monthly_debt', 'credit_score','household_income',
    #       'current_loan_amount', 'loan_type_risk_pct','loan_to_income_ratio',
    #                     'monthlydebt_to_income_ratio', 'bad_financial_condition']
    # print(all([i in df.columns for i in non_binary_fields]))
    # print(self.scaler.transform(df[non_binary_fields]))
    non_binary_fields = self.scaler.get_feature_names_out()
    df = pd.concat([pd.DataFrame(self.scaler.transform(df[non_binary_fields]), columns=non_binary_fields),
              df[[i for i in df.columns if i not in non_binary_fields]].reset_index(drop=True)], axis=1)
    # print('ttttttt')
    # print(df1)

    for i in ['years_of_credit_history', 'loan_type_risk_pct', 'years_in_current_job', 'loan_to_income_ratio',
              'monthly_debt', 'credit_score', 'household_income', 'current_loan_amount']:
      df[i+'_log'] = np.log10(df[i])
      df[i+'_exp'] = np.exp(df[i])

    return df[['current_loan_amount', 'credit_score', 'years_in_current_job', 'years_of_credit_history',
              'loan_type_risk_pct', 'loan_to_income_ratio', 'bad_financial_condition', 'long_term', 'home_mortgaged',
              'loan_type_other', 'social_class_middle_class', 'current_loan_amount_log', 'current_loan_amount_exp',
              'credit_score_log', 'credit_score_exp', 'household_income_exp', 'years_in_current_job_log',
              'years_in_current_job_exp', 'monthly_debt_log', 'monthly_debt_exp', 'years_of_credit_history_log',
              'years_of_credit_history_exp', 'loan_type_risk_pct_log', 'loan_type_risk_pct_exp',
              'loan_to_income_ratio_log', 'loan_to_income_ratio_exp']]


  # @staticmethod
  # def recommender_preprocessing(**fields):

  #   if fields['new_cust_index'] == True:
  #     fields['new_cust_index'] = 1
  #   else:
  #     fields['new_cust_index'] = 0

  #   if fields['cust_rank'] == 'Primary':
  #     fields['cust_rank'] = 1
  #   else:
  #     fields['cust_rank'] == 99

  #   if fields['activity_index'] == 'Active':
  #     fields['activity_index'] = 1
  #   else:
  #     fields['activity_index'] = 0

  #   if fields['birth_index']:
  #     fields['birth_index'] = 1
  #   else:
  #     fields['birth_index'] = 0


  #   df = pd.DataFrame(dict(zip(fields.keys(), [[i] for i in fields.values()])))
  #   to_dummy = {'emp_index': ['ex_employed', 'filial', 'not_employee', 'other'], 
  #                 'country': ['AE', 'AR', 'AT', 'AU', 'BE', 'BG', 'BO', 'BR', 'BY', 'CA', 'CH', 'CI', 'CL', 'CN', 'CO', 'CR', 'CU', 'CZ', 'DE', 'DK', 'DO', 'EC', 'ES', 'ET', 'FI', 'FR', 'GB', 'GT', 'GW', 'IE', 'IL', 'IT', 'JP', 'KE', 'KW', 'MA', 'MX', 'MZ', 'NL', 'OM', 'PE', 'PL', 'PT', 'PY', 'RO', 'RU', 'SA', 'SE', 'SG', 'SL', 'TG', 'TH', 'UA', 'US', 'UY', 'VE', 'ZA'], 
  #                 'cust_type': ['primary', 'former_primary', 'former_co-owner', 'potential', 'other'],
  #                 'relation_type': ['former', 'inactive', 'potential', 'other'],
  #                 'channel': ['KFA', 'KFC', 'KHE', 'KHQ', 'other'],
  #                 'cust_segment': ['individual', 'other', 'vip']}
  #   for field, vals in to_dummy.items():
  #     val_df = pd.DataFrame(dict(zip([field+'_'+i for i in vals],[[0]]*len(vals))))
  #     val = fields[field]
  #     df = pd.concat([df,val_df], axis=1)
  #     if (val != 'other') and (field not in ['cust_type','relation_type','channel', 'emp_index']):
  #       df.loc[0, field+'_'+val] = 1
  #     elif field in ['cust_type','relation_type','channel', 'emp_index']:
  #       df.loc[0, field+'_'+val] = 1
      
  #   df['cust_holding_period'] = np.round((datetime.datetime.now() - pd.to_datetime(df.cust_holding_date)).apply(lambda x: x.days/365), 2)
  #   df['cust_rank_lastday_period_till_date'] = np.round((datetime.datetime.now() - pd.to_datetime(df.cust_rank_lastday)).apply(lambda x: x.days/365), 2)
  #   df.loc[df.country=='ES', 'resident_of_spain'] = 1
  #   df.resident_of_spain = df.resident_of_spain.fillna(0)
  #   df.loc[df.birth_index=='S','foreigner'] = 1
  #   df.foreigner = df.foreigner.fillna(0)
  #   df.loc[df.gender=='M','male'] = 1
  #   df.male = df.male.fillna(0)
  #   # df = df.drop(list(to_dummy.keys())+['cust_holding_date','cust_rank_lastday','birth_index','gender'], axis=1)
  #   df = df[['age', 'new_cust_index', 'cust_seniority', 'cust_rank',
  #       'activity_index', 'household_income', 'cust_holding_period',
  #       'cust_rank_lastday_period_till_date', 'resident_of_spain',
  #       'foreigner', 'male', 'emp_index_ex_employed', 'emp_index_filial',
  #       'emp_index_not_employee', 'country_AE', 'country_AR', 'country_AT',
  #       'country_AU', 'country_BE', 'country_BG', 'country_BO',
  #       'country_BR', 'country_BY', 'country_CA', 'country_CH',
  #       'country_CI', 'country_CL', 'country_CN', 'country_CO',
  #       'country_CR', 'country_CU', 'country_CZ', 'country_DE',
  #       'country_DK', 'country_DO', 'country_EC', 'country_ES',
  #       'country_ET', 'country_FI', 'country_FR', 'country_GB',
  #       'country_GT', 'country_GW', 'country_IE', 'country_IL',
  #       'country_IT', 'country_JP', 'country_KE', 'country_KW',
  #       'country_MA', 'country_MX', 'country_MZ', 'country_NL',
  #       'country_OM', 'country_PE', 'country_PL', 'country_PT',
  #       'country_PY', 'country_RO', 'country_RU', 'country_SA',
  #       'country_SE', 'country_SG', 'country_SL', 'country_TG',
  #       'country_TH', 'country_UA', 'country_US', 'country_UY',
  #       'country_VE', 'country_ZA', 'cust_type_former_co-owner',
  #       'cust_type_former_primary', 'cust_type_other',
  #       'cust_type_potential', 'cust_type_primary', 'relation_type_former',
  #       'relation_type_inactive', 'relation_type_other',
  #       'relation_type_potential', 'channel_KFA', 'channel_KFC',
  #       'channel_KHE', 'channel_KHQ', 'channel_other',
  #       'cust_segment_individual', 'cust_segment_other',
  #       'cust_segment_vip']]      
  #   return np.array(df)

  # @staticmethod
  # def get_type(arr):
  #   lst = []
  #   for i in arr:
  #     tr = ', '.join([['Business Loan', 'Education Loan', 'Home Loan', 'Personal Loan', 'shopping loan',
  #                             'Vehicle Loan', 'wedding loan', 'other'][k] for k,l in enumerate(i) if l==1])
  #     lst.append(tr)
  #   return lst


  risk_predict = lambda self, **fields: ['Will default','Will pay'][self.risk_model.predict(self.risk_preprocessing(**fields))[0]]
  # recommend = lambda self, **fields: prediction.get_type(self.recommender.predict(prediction.recommender_preprocessing(**fields)))[0]