import pandas as pd
import numpy as np
import os
import pickle
import warnings
from collections import Counter

warnings.filterwarnings( 'ignore' )

# Create MIMIC-III dataset
def create_MIMIC3_ICU(args):
    time_gap_hours = args.time_gap_hours
    pred_window_hours = args.pred_window_hours
    timegap = pd.Timedelta(time_gap_hours, unit="h") 
    pred_window = pd.Timedelta(pred_window_hours, unit="h") 

    patient_path = os.path.join(args.input_path, 'mimic3', 'PATIENTS.csv')
    icustay_path = os.path.join(args.input_path, 'mimic3', 'ICUSTAYS.csv')
    dx_path = os.path.join(args.input_path, 'mimic3', 'DIAGNOSES_ICD.csv')
    ad_path = os.path.join(args.input_path, 'mimic3', 'ADMISSIONS.csv')

    patients = pd.read_csv(patient_path)
    icus = pd.read_csv(icustay_path)
    add = pd.read_csv(ad_path)
    dx = pd.read_csv(dx_path)

    print('length of PATIENTS.csv  : ', len(patients))
    print('length of ICUSTAYS.csv  : ', len(icus))
    print('length of DIAGNOSIS_ICD.csv  : ', len(icus))
    print('length of ADMISSION.csv  : ', len(add))


    temp = icus[(icus['FIRST_CAREUNIT'] == icus['LAST_CAREUNIT'])]
    if args.icu_type =='micu':
        temp = temp[temp.LAST_CAREUNIT == 'MICU']     
    #->For MICU

    temp = temp.drop(columns=['ROW_ID'])
    temp['INTIME'] = pd.to_datetime(temp['INTIME'], infer_datetime_format=True)
    temp['OUTTIME'] = pd.to_datetime(temp['OUTTIME'], infer_datetime_format=True)


    patients['DOB'] = pd.to_datetime(patients['DOB'], infer_datetime_format=True)
    patients = patients.drop(columns=['ROW_ID'])

    small_patients = patients[patients.SUBJECT_ID.isin(temp.SUBJECT_ID)]
    temp = temp.merge(small_patients, on='SUBJECT_ID', how='left')


    datediff = np.array(temp.INTIME.dt.date) - np.array(temp.DOB.dt.date)
    age = np.array([x.days // 365 for x in datediff])
    temp['age'] = age
    temp = temp[temp.age >= 18]
    print('length of temp  :', len(temp))

    readmit = temp.groupby('HADM_ID')['ICUSTAY_ID'].count()
    readmit_labels = (readmit > 1).astype('int64').to_frame().rename(columns={'ICUSTAY_ID':'readmission'})
    print('readmission value counts :', readmit_labels.value_counts())

    small_temp = temp.loc[temp.groupby('HADM_ID').INTIME.idxmin()]
    small_temp= pd.merge(small_temp.reset_index(drop=True), add[['HADM_ID','DISCHARGE_LOCATION', 'DEATHTIME', 'DISCHTIME']], how='left', on='HADM_ID')
    readmission_cohort = small_temp.join(readmit_labels, on='HADM_ID')
    cohort = readmission_cohort
    cohort['DEATHTIME'] = pd.to_datetime(cohort['DEATHTIME'], infer_datetime_format=True)
    cohort['DISCHTIME'] = pd.to_datetime(cohort['DISCHTIME'], infer_datetime_format=True)
    dead = cohort[~pd.isnull(cohort.DEATHTIME)].copy()
    # mortality labeling
    dead_labels = (
        (dead.DEATHTIME <= dead.OUTTIME) & 
        (dead.DEATHTIME >= dead.INTIME + pd.Timedelta(12, unit="h") ) &
        (dead.DEATHTIME <= dead.INTIME + pd.Timedelta(12, unit="h") + timegap + pred_window )
    )
    dead_labels = dead_labels.astype('int64')
    dead['mortality'] = np.array(dead_labels)

    #prediction window labels 
    in_icu_dead_labels = ((dead.DEATHTIME > dead.INTIME) & (dead.DEATHTIME <= dead.OUTTIME))
    in_icu_dead_labels  = in_icu_dead_labels.astype('int64')
    dead['in_icu_mortality'] = np.array(in_icu_dead_labels)


    cohort = pd.merge(cohort.reset_index(drop=True), dead[['ICUSTAY_ID','mortality','in_icu_mortality']], on='ICUSTAY_ID', how='left').reset_index(drop=True)
    cohort['mortality'] = cohort['mortality'].fillna(0)
    cohort['in_icu_mortality'] = cohort['in_icu_mortality'].fillna(0)
    cohort = cohort.astype({'mortality': int})
    cohort = cohort.astype({'in_icu_mortality': int})
    cohort['los_3day'] = (cohort['LOS'] > 3.).astype('int64')
    cohort['los_7day'] = (cohort['LOS'] > 7.).astype('int64')

    # time_gap

    cohort12h = cohort[cohort['LOS'] > (0.5 + time_gap_hours/24)].reset_index(drop=True)
    cohort12hDx = dx[dx.HADM_ID.isin(cohort12h.HADM_ID)]
    diagnosis = cohort12hDx.groupby('HADM_ID')['ICD9_CODE'].apply(list).to_frame()
    tempdf = cohort12h.join(diagnosis, on='HADM_ID')

    tempdf.loc[(tempdf.DISCHARGE_LOCATION =='DEAD/EXPIRED') &(tempdf.in_icu_mortality !=1),'in_hospital_mortality'] = 1

    tempdf['in_hospital_mortality']= tempdf['in_hospital_mortality'].fillna(0).astype('int64')

    tempdf.loc[(tempdf.in_icu_mortality ==1), 'DISCHARGE_LOCATION'] = 'IN_ICU_MORTALITY'
    tempdf.loc[(tempdf.in_hospital_mortality ==1), 'DISCHARGE_LOCATION'] = 'IN_HOSPITAL_MORTALITY'

    print(tempdf.DISCHARGE_LOCATION.value_counts()/len(tempdf)*100)
    print(tempdf.DISCHARGE_LOCATION.astype('category').cat.categories)

    tempdf['final_acuity'] = tempdf.DISCHARGE_LOCATION.astype('category').cat.codes

    discharge =  (
        (tempdf['OUTTIME'] <= tempdf.INTIME + pd.Timedelta(12, unit="h") + pred_window)
    )

    tempdf.loc[discharge, 'imminent_discharge'] = tempdf[discharge]['DISCHARGE_LOCATION'] 

    tempdf.loc[~discharge, 'imminent_discharge'] = 'No Discharge'

    tempdf.loc[(
        (tempdf['imminent_discharge']=='IN_HOSPITAL_MORTALITY') |
        (tempdf['imminent_discharge']=='IN_ICU_MORTALITY')
    ), 'imminent_discharge'] = 'No Discharge'


    print(tempdf['imminent_discharge'].value_counts()/len(tempdf)*100)
    print(tempdf['imminent_discharge'].astype('category').cat.categories)

    tempdf['imminent_discharge'] = tempdf['imminent_discharge'].astype('category').cat.codes

    cohort = tempdf.drop(columns=['in_hospital_mortality', 'in_icu_mortality', 'DISCHARGE_LOCATION'])

    #diagnosis label
    ccs_dx = pd.read_csv('/home/edlab/ghhur/Pretrained_DescEmb/raw_file/mimic3/ccs_multi_dx_tool_2015.csv')
    ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:].str[:-1].str.replace(" ", "")
    ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:].str[:-1]
    level1 = {}
    for x, y in zip(ccs_dx["'ICD-9-CM CODE'"], ccs_dx["'CCS LVL 1'"]):
        level1[x] = y
    
    dx1_list = []
    for idx, dxx in enumerate(cohort['ICD9_CODE']):
        one_list = []
        for dx in dxx:
            if dx not in level1.keys():
                continue
            dx1 = level1[dx]
            one_list.append(dx1)
        dx1_list.append(list(set(one_list)))
    cohort['diagnosis'] = pd.Series(dx1_list)
    cohort = cohort[cohort['diagnosis'] !=float].reset_index(drop=True)
    dx1_length = [len(i) for i in dx1_list]
    print("average length: ", np.array(dx1_length).mean())
    print('dx freqeuncy', np.bincount(dx1_length))
    print("max length: ", np.array(dx1_length).max())
    print("min length: ", np.array(dx1_length).min())

    save_path = os.path.join(args.save_path, 'preprocess',args.icu_type, 'time_gap_'+str(args.time_gap_hours),'mimic3_cohort.pkl')
    print(f'The final MIMIC3 cohort pickle is saved at: {save_path}')
    cohort.to_pickle(save_path)



# Create eICU dataset
def create_eICU_ICU(args):
    
    time_gap_hours = args.time_gap_hours
    pred_window_hours = args.pred_window_hours
    timegap = pd.Timedelta(time_gap_hours, unit="h") 
    pred_window = pd.Timedelta(pred_window_hours, unit="h")

    patient_path = os.path.join(args.input_path, 'eicu', 'patient.csv')
    patient_df = pd.read_csv(patient_path)
    dx_path = os.path.join(args.input_path, 'eicu', 'diagnosis.csv')
    dx = pd.read_csv(dx_path)

    print('Unique patient unit stayid : ', len(set(patient_df.patientunitstayid)))

    micu = patient_df
    if args.icu_type =='micu':
        micu = patient_df[patient_df.unittype == 'MICU']  
    #-> For Total ICU

    null_index =micu[micu['age'].isnull()==True].index
    micu.loc[null_index, 'age'] = 1
    micu = micu.replace('> 89', 89)

    micu.loc[:, 'age'] = micu.loc[:, 'age'].astype('int')
    micuAge = micu[micu.age >= 18]

    readmit = micuAge.groupby('patienthealthsystemstayid')['patientunitstayid'].count()
    readmit_labels = (readmit > 1).astype('int64').to_frame().rename(columns={'patientunitstayid':'readmission'})

    firstIcus = micuAge.loc[micuAge.groupby('patienthealthsystemstayid').hospitaladmitoffset.idxmax()]
    readmission_cohort = firstIcus.join(readmit_labels, on='patienthealthsystemstayid')

    cohort = readmission_cohort

    cohort['mortality'] = (
        (cohort['unitdischargeoffset'] < (60*(12 + time_gap_hours+ pred_window_hours))) &
        (cohort['unitdischargeoffset'] > (60*(12))) &
        (cohort['unitdischargestatus'] == 'Expired')
    ).astype('int64')
        
    cohort['in_icu_mortality'] = (cohort['unitdischargestatus'] == 'Expired').astype('int64')
    cohort['in_hospital_mortality'] = (
        (cohort['hospitaldischargestatus'] == 'Expired').astype('int64') &
        (cohort['in_icu_mortality'] != 1)
    ).astype('int64')

    cohort.loc[(cohort.in_icu_mortality ==1), 'hospitaldischargelocation'] = 'IN_ICU_MORTALITY'
    cohort.loc[(cohort.in_hospital_mortality ==1), 'hospitaldischargelocation'] = 'IN_HOSPITAL_MORTALITY'

    cohort['final_acuity'] = cohort.hospitaldischargelocation.astype('category').cat.codes
    d = dict(enumerate(cohort.hospitaldischargelocation.astype('category').cat.categories))
    print('final acuity class category', d.items())

    print(cohort.hospitaldischargelocation.value_counts()/len(cohort)*100)
    print(cohort.hospitaldischargelocation.astype('category').cat.categories)

    discharge =  (
        (cohort['unitdischargeoffset'] < (60*(12 + time_gap_hours+ pred_window_hours)))
    )

    cohort.loc[discharge, 'imminent_discharge'] = cohort[discharge]['hospitaldischargelocation'] 

    cohort.loc[~discharge, 'imminent_discharge'] = 'No Discharge'

    cohort.loc[(
        (cohort['imminent_discharge']=='IN_HOSPITAL_MORTALITY') |
        (cohort['imminent_discharge']=='IN_ICU_MORTALITY')
    ), 'imminent_discharge'] = 'No Discharge'


    print(cohort['imminent_discharge'].value_counts()/len(cohort)*100)
    print(cohort['imminent_discharge'].astype('category').cat.categories)

    cohort['imminent_discharge'] =cohort['imminent_discharge'].astype('category').cat.codes
    # d = dict(enumerate(cohort['imminent_discharge'].catcategories))
    # print('imminent_discharge class category', d.items())

    
    cohort['losday'] = (cohort['unitdischargeoffset'].astype('float') / (24.*60.))
    cohort['los_3day'] = (cohort['losday'] > 3.).astype('int64')
    cohort['los_7day'] = (cohort['losday'] > 7.).astype('int64')

    cohort12h = cohort[cohort['unitdischargeoffset'] > (60*(12 + time_gap_hours))]
    cohort12hDx = dx[dx.patientunitstayid.isin(cohort12h.patientunitstayid)]
    diagnosis = cohort12hDx.groupby('patientunitstayid')['diagnosisstring'].apply(list).to_frame()

    dxDict = dict(enumerate(cohort12hDx.groupby('diagnosisstring').count().index))
    dxDict = dict([(v,k) for k,v in dxDict.items()])
    print('diganosis unique : ', len(dxDict))

    tempdf = cohort12h.join(diagnosis, on='patientunitstayid') 
    tempdf = tempdf.drop(columns=['in_hospital_mortality', 'in_icu_mortality', 'hospitaldischargelocation'])
    
    cohort_ei = tempdf.copy().reset_index(drop=True)
    cohort_ei = eicu_diagnosis_label(cohort_ei)
    cohort_ei = cohort_ei[cohort_ei['diagnosis'] !=float].reset_index(drop=True)
    cohort_ei = cohort_ei.reset_index(drop=True)

    
    save_path = os.path.join(args.save_path, 'preprocess', args.icu_type, 'time_gap_'+str(args.time_gap_hours),'eicu_cohort.pkl')
    print(f'The final eicu cohort pickle is saved at: {save_path}')
    cohort_ei.to_pickle(save_path)

def eicu_diagnosis_label(eicu_cohort):
    ccs_dx = pd.read_csv('/home/edlab/ghhur/Pretrained_DescEmb/raw_file/mimic3/ccs_multi_dx_tool_2015.csv')
    ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:].str[:-1].str.replace(" ", "")
    ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:].str[:-1]
    level1 = {}
    for x, y in zip(ccs_dx["'ICD-9-CM CODE'"], ccs_dx["'CCS LVL 1'"]):
        level1[x] = y 

    eicu_dx_df = eicu_cohort.dropna(subset=['diagnosisstring']).copy().reset_index(drop=True)
    eicu_diagnosis_list = []
    for x in eicu_dx_df['diagnosisstring']:
        eicu_diagnosis_list.extend(x)
    eicu_dx_unique = list(set(eicu_diagnosis_list))
    eicu_dx = pd.read_csv('/home/edlab/ghhur/Pretrained_DescEmb/raw_file/eicu/diagnosis.csv')

    # eicu_dx all diagnosis status
    eicu_dx_list = list(eicu_dx['icd9code'].values)
    eicu_dx_list = [x for x in eicu_dx_list if x != 'nan' and type(x) != float]
    eicu_dx_list = [y.strip().replace('.', '') for x in eicu_dx_list for y in x.split(',')]
    eicu_ids = list(eicu_dx_df['patientunitstayid'].values)

    # drop the icd9code NaN for right now
    eicu_dx = eicu_dx.dropna(subset=['icd9code']).copy().reset_index(drop=True)

    # make diagnosisstring - ICD9 code dictionary
    diagnosisstring_code_dict = {}
    key_error_list = []

    for index, row in eicu_dx.iterrows():
        diagnosis_string = row['diagnosisstring']
        icd9code = row['icd9code']
        icd9code = icd9code.split(',')[0].replace('.','')
        try: 
            eicu_level1 = level1[icd9code]
            diagnosisstring_code_dict[diagnosis_string] = eicu_level1
        except KeyError:
            key_error_list.append(diagnosis_string)  

    # Check key error list
    key_error_list = list(set(key_error_list))
    print('Number of diagnosis with only ICD 10 code: {}'.format(len(key_error_list)))

    
    # icd10 to icd9 mapping csv file
    icd10_icd9 = pd.read_csv('/home/edlab/ghhur/Pretrained_DescEmb/raw_file/eicu/icd10cmtoicd9gem.csv')

    # make icd10 - icd9 dictionary
    icd10_icd9_dict = {}
    for x, y in zip(icd10_icd9['icd10cm'], icd10_icd9['icd9cm']):
        icd10_icd9_dict[x] = y

    # map icd10 to icd9 code
    two_icd10_code_list = []
    icd10_key_error_list = []
    for i in range(len(key_error_list)):
        icd10code = eicu_dx[eicu_dx['diagnosisstring'] == key_error_list[i]]['icd9code'].values[0].split(',')
        if len(icd10code) >= 2:
            two_icd10_code_list.append(key_error_list[i])
            continue
            
        elif len(icd10code) == 1:
            icd10code = icd10code[0].replace('.','')
            try:
                icd9code = icd10_icd9_dict[icd10code]
                diagnosisstring_code_dict[key_error_list[i]] = level1[icd9code]
            except KeyError:
                icd10_key_error_list.append(key_error_list[i])
    print('Number of more than one icd10 codes : {}'.format(len(two_icd10_code_list)))
    print('Number of icd10key_error_list : {}'.format(len(icd10_key_error_list)))


    # deal with more than one ICD10 code
    class_list = ['6', '7', '6', '7', '2', '6', '6', '7', '6', '6','6']
    for i in range(11):
        diagnosisstring_code_dict[two_icd10_code_list[i]] = class_list[i]

    # fill in the blank!
    have_to_find = []
    already_in = []
    for i in range(len(eicu_dx_unique)):
        single_dx = eicu_dx_unique[i]
        try:
            oneoneone = diagnosisstring_code_dict[single_dx]
            already_in.append(single_dx)
        except KeyError:
            have_to_find.append(single_dx)
    print('Number of dx we have to find...{}'.format(len(have_to_find)))

    # one hierarchy above
    have_to_find2 = []
    for i in range(len(have_to_find)):
        s = "|".join(have_to_find[i].split('|')[:-1])
        try:
            depth1_code = diagnosisstring_code_dict[s]
            diagnosisstring_code_dict[have_to_find[i]] = depth1_code
        except KeyError:
            have_to_find2.append(have_to_find[i])
    print('Number of dx we have to find...{}'.format(len(have_to_find2)))

    # hierarchy below
    dict_keys = list(diagnosisstring_code_dict.keys())

    have_to_find3 = []
    for i in range(len(have_to_find2)):
        s = have_to_find2[i]
        dx_list = []
        for k in dict_keys:
            if k[:len(s)] == s:
                dx_list.append(diagnosisstring_code_dict[k])
        
        dx_list = list(set(dx_list))
        if len(dx_list) == 1:
            diagnosisstring_code_dict[s] = dx_list[0]
        else:
            have_to_find3.append(s)
            
    print('Number of dx we have to find...{}'.format(len(have_to_find3)))

    # hierarchy abovs
    dict_keys = list(diagnosisstring_code_dict.keys())
    have_to_find4 = []

    for i in range(len(have_to_find3)):
        s = "|".join(have_to_find3[i].split('|')[:-1])
        dx_list = []
        for k in dict_keys:
            if k[:len(s)] == s:
                dx_list.append(diagnosisstring_code_dict[k])
                
        dx_list = list(set(dx_list))
        if len(dx_list) == 1:
            diagnosisstring_code_dict[have_to_find3[i]] = dx_list[0]
        else:
            have_to_find4.append(have_to_find3[i])

    print('Number of dx we have to find...{}'.format(len(have_to_find4)))

    for t in range(4):
        c = -t-1
    dict_keys = list(diagnosisstring_code_dict.keys())
    have_to_find_l = []
    for i in range(len(have_to_find4)):
        s = have_to_find4[i]
        s = "|".join(s.split("|")[:c])
        dx_list =[]
        for k in dict_keys:
            if k[:len(s)] == s:
                dx_list.append(diagnosisstring_code_dict[k])
        dx_list2 = list(set(dx_list))
        if len(dx_list2) > 1:
            cnt = Counter(dx_list)
            mode = cnt.most_common(1)
            diagnosisstring_code_dict[have_to_find4[i]] = mode[0][0]
        else:
            have_to_find_l.append(have_to_find4[i])
    del(have_to_find4)
    have_to_find4 = have_to_find_l.copy()
    print('Number of dx we have to find...{}'.format(len(have_to_find4)))

    dx_depth1 = []
    dx_depth1_unique = []
    solution = lambda data: [x for x in set(data) if data.count(x) != 1]

    for ICD_list in eicu_dx_df['diagnosisstring']:
        single_list = list(pd.Series(ICD_list).map(diagnosisstring_code_dict))
        dx_depth1.append(single_list)
        dx_depth1_unique.append(list(set(single_list)))
    eicu_dx_df['diagnosis'] = pd.Series(dx_depth1_unique)

    return eicu_dx_df



# Create MIMIC4 dataset
def create_MIMIC4_ICU(args):
    time_gap_hours = 2
    pred_window_hours = 24
    timegap = pd.Timedelta(time_gap_hours, unit="h") 
    pred_window = pd.Timedelta(pred_window_hours, unit="h") 

    icu = pd.read_csv(os.path.join(args.input_path, 'mimic4', 'icustays.csv'))
    adm = pd.read_csv(os.path.join(args.input_path, 'mimic4', 'admissions.csv'))
    pat = pd.read_csv(os.path.join(args.input_path, 'mimic4', 'patients.csv'))
    # dx = pd.read_csv(os.path.join(args.input_path, 'mimic4', 'diagnoses_icd.csv'))
    dx = convert_icd_mimic4()
    # dx 를 10에서 9로 바꾸고
    # list 만들어서 convert해주고ㅛ

    def columns_upper(df):
        df.columns = [x.upper() for x in df.columns]

    for df in [icu, adm, pat, adm, dx]:
        columns_upper(df)
    columns_upper(dx)
    # ICU
    icu.rename(columns= {'STAY_ID': 'ICUSTAY_ID'}, inplace=True)

    # 입원기간 - 최소 12시간 초과
    temp = icu[icu.LOS > (0.5 + time_gap_hours/24)]

    adm_sd = adm[['HADM_ID', 'DEATHTIME']].copy() #sd stands for subject_id, deathtime
    adm_sd.sort_values('DEATHTIME', ascending= True, inplace= True)

    k = adm_sd.groupby('HADM_ID')


    adm_sd = k.first()
    adm_sd.reset_index(inplace= True)
    adm_sd.HADM_ID.is_unique # DEATHTIME만 merge하기 위해서 unique HADM_ID - DEATHTIME pair 만듦

    # adding DEATHTIME columns 
    temp = temp.merge(adm_sd, on= 'HADM_ID')

    # 나이 - 18세 이상
    # 문제점: 300살 짜리 생김 -> 초고령자 deidentification
    temp = temp.merge(pat[['SUBJECT_ID','ANCHOR_AGE', 'ANCHOR_YEAR']], on= 'SUBJECT_ID')

    # 만나이로 계산은 못함... 
    temp.INTIME = pd.to_datetime(temp.INTIME)
    temp['age'] = temp.INTIME.dt.year - temp.ANCHOR_YEAR + temp.ANCHOR_AGE
    temp = temp[temp.age >= 18]

    # Readmission: ICU가 갯수가 2개이상인 경우

    df = temp[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].groupby('HADM_ID').count()#.sort_values(by = 'ICUSTAY_ID')
    df['readmission'] = df['ICUSTAY_ID'].apply(lambda x: 1 if x > 1 else 0)
    df.reset_index(inplace= True)

    temp = temp.merge(df[['HADM_ID', 'readmission']], on= 'HADM_ID')

    # 한 환자가 여러 ICU 가지고 있는 경우 - 제일 첫번째 ICU만 가져옴.
    # 소팅 기준 확인: 첫번째 ICU란? - groupby 해봤을 때 intime 기준아님.
    # intime기준 sorting

    temp.sort_values('INTIME', ascending= True, inplace= True)
    
    temp = temp.groupby('HADM_ID').first() # 제일 첫번쨰 ICU만 가져옴.
    temp['HADM_ID'] = temp.index
    temp.reset_index(drop=True, inplace= True)
    temp = pd.merge(temp, adm[['HADM_ID','DISCHARGE_LOCATION']], on='HADM_ID', how='left').reset_index(drop=True)
    # Los > 3day: Length of stay 3일 이상 
    # Los > 7day: Length of stay 7일 이상
    
    temp['los_3day'] = temp['LOS'].apply(lambda x: 1 if x >= 3 else 0)
    temp['los_7day'] = temp['LOS'].apply(lambda x: 1 if x >= 7 else 0)

    # adding (12h_obs, 24h_obs) columns
    temp.INTIME = pd.to_datetime(temp.INTIME)
    temp['12h_obs'] = temp.INTIME + pd.Timedelta(pd.offsets.Hour(12))
    temp['24h_obs'] = temp.INTIME + pd.Timedelta(pd.offsets.Hour(24))

    # Mortality
    temp.INTIME, temp.OUTTIME, temp.DEATHTIME = pd.to_datetime(temp.INTIME), pd.to_datetime(temp.OUTTIME), pd.to_datetime(temp.DEATHTIME)

    mortality = (
         (temp.DEATHTIME >= temp.INTIME + pd.Timedelta(12, unit="h") ) &
        (temp.DEATHTIME <= temp.INTIME + pd.Timedelta(12, unit="h") + timegap + pred_window)
    )
    temp['mortality'] = mortality.astype(int)
    in_icu_mortality = (temp.INTIME < temp.DEATHTIME) & (temp.OUTTIME >= temp.DEATHTIME)

    
    temp['in_icu_mortality'] = in_icu_mortality.astype(int)

    temp.loc[(temp.DISCHARGE_LOCATION =='DIED') &(temp.in_icu_mortality !=1),'in_hospital_mortality'] = 1
    temp['in_hospital_mortality']= temp['in_hospital_mortality'].fillna(0).astype('int64')

    temp.loc[(temp.in_icu_mortality ==1), 'DISCHARGE_LOCATION'] = 'IN_ICU_MORTALITY'
    temp.loc[(temp.in_hospital_mortality ==1), 'DISCHARGE_LOCATION'] = 'IN_HOSPITAL_MORTALITY'
    print(temp.DISCHARGE_LOCATION.value_counts()/len(temp)*100)
    print(temp.DISCHARGE_LOCATION.astype('category').cat.categories)

    temp['final_acuity'] = temp.DISCHARGE_LOCATION.astype('category').cat.codes
    d = dict(enumerate(temp.DISCHARGE_LOCATION.astype('category').cat.categories))
    print('final acuity class category', d.items())
    discharge =  (
        (temp['OUTTIME'] <= temp.INTIME + pd.Timedelta(12, unit="h") + timegap + pred_window)
    )
    temp.loc[discharge, 'imminent_discharge'] = temp[discharge]['DISCHARGE_LOCATION'] 
    temp.loc[~discharge, 'imminent_discharge'] = 'No Discharge'

    temp.loc[(
        (temp['imminent_discharge']=='IN_HOSPITAL_MORTALITY') |
        (temp['imminent_discharge']=='IN_ICU_MORTALITY')
    ), 'imminent_discharge'] = 'No Discharge'

    
    print(temp['imminent_discharge'].value_counts()/len(temp)*100)
    print(temp['imminent_discharge'].astype('category').cat.categories)

    temp['imminent_discharge'] = temp['imminent_discharge'].astype('category').cat.codes
    #d = dict(enumerate(temp['imminent_discharge'].cat.categories))
    #print('imminent discharge class category', d.items())

    cohort = temp.drop(columns=['in_hospital_mortality', 'in_icu_mortality', 'DISCHARGE_LOCATION'])
    
    # # debug
    # #save
    # with open('/home/mjlee/playground/cohort_debug.pkl', 'wb') as f:
    #     pickle.dump(cohort, f, pickle.HIGHEST_PROTOCOL)
    # print('a pickle for debugging is saved! ')
    # load
    # with open('/home/mjlee/playground/cohort_debug.pkl', 'rb') as f:
    #     cohort = pickle.load(f)
    # print('a pickle for debugging is loaded! ') 

    # cohort12h = cohort[cohort['LOS'] > (0.5 + time_gap_hours/24)].reset_index(drop=True)
    cohort12hDx = dx[dx.HADM_ID.isin(cohort.HADM_ID)]
    diagnosis = cohort12hDx.groupby('HADM_ID')['ICD_CODE_CONVERTED'].apply(list).to_frame()
    cohort = cohort.join(diagnosis, on='HADM_ID')

    #diagnosis label
    ccs_dx = pd.read_csv('/home/edlab/ghhur/Pretrained_DescEmb/raw_file/mimic3/ccs_multi_dx_tool_2015.csv')
    ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:].str[:-1].str.replace(" ", "")
    ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:].str[:-1]
    level1 = {}
    for x, y in zip(ccs_dx["'ICD-9-CM CODE'"], ccs_dx["'CCS LVL 1'"]):
        level1[x] = y

    dx1_list = []

    for idx, dxx in enumerate(cohort['ICD_CODE_CONVERTED']):
        one_list = []
        # print(idx, dxx)
        if type(dxx) != float: # dxx==NaN case
            for dx in dxx:
                if dx not in level1.keys():
                    continue

                dx1 = level1[dx]
                one_list.append(dx1)
        dx1_list.append(list(set(one_list)))
    cohort['diagnosis'] = pd.Series(dx1_list)
    cohort = cohort[cohort['diagnosis'] !=float].reset_index(drop=True)
    dx1_length = [len(i) for i in dx1_list]
    print("average length: ", np.array(dx1_length).mean())
    print('dx freqeuncy', np.bincount(dx1_length))
    print("max length: ", np.array(dx1_length).max())
    print("min length: ", np.array(dx1_length).min())

    save_path = os.path.join(args.save_path, 'preprocess', args.icu_type, 'time_gap_'+str(args.time_gap_hours),'mimic4_cohort.pkl')
    print(f'The final MIMIC4 cohort pickle is saved at: {save_path}')
    cohort.to_pickle(save_path)
    

def convert_icd_mimic4():
    # load dx from MIMIC4
    src_path = '/home/edlab/ghhur/Pretrained_DescEmb/raw_file/mimic4/'
    dx = pd.read_csv(os.path.join(src_path, 'diagnoses_icd.csv'))

    # load mapping from CMS (2018 ver.)
    file_map_10_cms = '/home/data_storage/UMLS/diagnosis_gems_2018/2018_I10gem.txt'
    map_10_cms = pd.read_csv(file_map_10_cms, sep='\s+', header=None, names=['icd10cm', 'icd9cm', 'flags'])

    dx_icd_10 = dx[dx.icd_version == 10].icd_code

    unique_elem_no_map = set(dx_icd_10) - set(map_10_cms.icd10cm)
    num_elem_no_map = dx_icd_10.apply(lambda x: x in unique_elem_no_map).sum()

    # make map_cms (when icd10 code is in the map)
    map_cms = dict(zip(map_10_cms.icd10cm, map_10_cms.icd9cm))

    # make map_manual (when icd10 code is NOT in the map)
    map_manual = dict.fromkeys(unique_elem_no_map, 'NaN')
    map_unique_icd10 = set(map_10_cms.icd10cm)
    str_icd10cm = ' '.join(map_10_cms.icd10cm.to_list())

    for code_10 in map_manual:
        for i in range(len(code_10), 0, -1):
            tgt_10 = code_10[:i]
            if tgt_10 in str_icd10cm:
                tgt_9 = map_10_cms[map_10_cms.icd10cm.str.contains(tgt_10)].icd9cm.mode().iloc[0]
                map_manual[code_10] = tgt_9 
                break

    def icd_convert(icd_version, icd_code):
        if icd_version == 9:
            return icd_code

        elif icd_code in map_cms:
            return map_cms[icd_code]

        elif icd_code in map_manual:
            return map_manual[icd_code]

    # convert MIMIC4's ICD10 to ICD9
    dx['icd_code_converted'] = dx.apply(lambda x: icd_convert(x['icd_version'], x['icd_code']), axis=1)
    
    return dx