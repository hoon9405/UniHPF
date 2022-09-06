# UniHPF : Universal Healthcare Predictive Framework with Zero Domain Knowledge

UniHPF is a universal healthcare predictive framework, which requires no medical domain knowledge and minimal preprocessing for multiple prediction tasks. Given any time-series EHR, 		 	 	 			
					
Our framework presents a method for embedding any form of EHR systems for prediction tasks with- out requiring domain-knowledge-based pre-processing, such as medical code mapping and feature selection.  
				
This repository provides official Pytorch code to implement UniHPF, a universal healthcare predictive framework.

# Supplementary Results
## Detailed Preprocessing information	
More information about how datasets were created is provided in this section.	
### Table selection
* For each patient, three sources with different “event types”(lab tests, prescription, and infusion) are preprocessed as input for a predictive model. Table S1 lists csv filenames with each event type.
* Note that MIMIC-III files ’INPUTEVENTS MV’ and ’INPUTEVENTS CV’,  are merged and named as INPUTEVENTS.
* File names for each data sources and tables are described below.

|    | **MIMIC-III**     | **eICU**         | **MIMIC-IV**      |
|------------------|-------------------|------------------|--------------------|
| **Lab test**     | LABEVENTS.csv     | lab.csv          | labevents.csv      |
| **Prescription** | PRESCRIPTIONS.csv | medication.csv   | prescriptions.csv  |
| **Infusion**     | INPUTEVENTS.csv   | infusionDrug.csv | inputevents.csv    |

### Patient cohort setup
Then, we prepare patient cohorts from MIMIC-III, MIMIC-IV and eICU databsets based on the following standards for comparability: Patients in the intensive care unit (ICU) who are (1) over 18 and (2) have been there for more than 24 hours. We only use the first ICU stay for patients who have multiple stays. Also, we restrict samples to the first 12 hours of data and drop any stays with fewer than five observed events. Lastly,  we eliminate events with lower frequency of main columns (drug_name, ITEMID, … ).

### Convert EHR table to input sequence
Here, we will explain our pre-process algorithm which enables us to deal with any EHR table, converting them into the same input configuration for UniHPF.
The process explanation is represented below.
1. First, replace code features to description if the definition table exists in the EHR source set, which the definition table has features as key and description as value. (e.g. MIMIC-III D_ITEMS.csv)  
2. Remove columns whose data type is integer except columns which have categorical values (e.g. number of unique features <50). 
3. Select the associated timestamp column which is most relevant to the point of occurrence and drop the other timestamp columns.
4. Convert all features as string type and tokenize them with “bio-clinical-bert” tokenizer except associated timestamp columns.
5. For numeric values in feature, split them digit by digit before being tokenized and apply digit-place embedding (DPE) following the value embedding method from DescEmb, which assigns a special token for each digit place.
6. Descriptions corresponding to each event are listed in the order of event type, feature name, and feature value.   
7. For time stamps, the time interval between the corresponding event and the next event is used as the time feature. 8. At this time, we follow Rajikomar method to deal with continuous values,  quantizing them into discretized features. So, the time interval is bucketed into 20 separations within the entire time interval and converted to special tokens.
9. Next, create a class of type token corresponding to the event type, feature name, and feature value. 
10. Lists events in order based on timestamp. Note that these type tokens are used as indicating each sub-word token type (event type, feature name, feature value). 
11. This type token sequence is added to event input with sinusoidal positional embedding.
12. Finally, prepare an input dataset with a shape as (N, S, W), where N is a number of icu stay, S is maximum length of events, and W is maximum sub-word length for each event.  			

### Datasets preparation for each model
To prepare a dataset for each model used in the experiment, we prepare the dataset following additional steps.

1. Feature selection
  - We prepare two versions of the dataset,  feature selection version and without feature selection version (using Entire EHR).
  - This was to compare the case with and without the conventional feature selection process, and in the case of SAnD and DescEmb, the feature selected dataset is used.
  - Feature selection criteria follows DescEmb, which are using information corresponding to medical code, numerical value,  unit of measurement.

2. Conventional embedding method
  - Each feature is coded based on unique text.
  - Before converting feature text into unique code, continuous values are buckettized after being grouped by each ITEMID. 
  - For categorical features, preprocessing is performed separately on categorical code.
  - Feature names (columns) are also converted as codes. 
3. Flatten structure
  - The hierarchical form (N, S, W) of input data is reshaped into the shape of (B, SxW). 
  - After removing the pad in each W, flattened input shape is changed to (N, S*) where S* indicates flattened input without pad.
  - SAnD* used this flattened dataset as input.
  - The ablation study results for flatten and hierarchical are below.

## Prediction tasks details
- Following the [Benchmark](https://dl.acm.org/doi/10.1145/3450439.3451877, "benchmark paper link") paper, prediction tasks are well defined. 
- Medical event information from ICU admission to 12 hours duration is used, and TimeGAP is given 12 hours for all tasks.
- The rolling type task (mortality, imminent discharge) is applied only for the first rolling point(similar to static type task), and the prediction window was given at 48hr.
- In the case of diagnosis, we tried to group CCS into 18 diagnosis classes based on CCS ontology. MIMIC-III, MIMIC-IV and eICU used “Diagnosis.csv”, “diagnoses_icd.csv” and “diagnosis.csv” respectively.
- Detailed label definition in the code.

| **Target**    | **MIMIC**                                                                                           | **eICU**                                                                                  |
|-----------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Mortality | ‘unitDischargeStatus’==‘Expired’ and (Timegap< discharge time -INTIME   < prediction window)    | (‘DOD HOSP’ not null) and (Timegap< discharge time -INTIME <   prediction window)     |
| LOS3      | LOS >3                                                                                          | ‘unitDischargeOffset’ >3*24*60'                                                       |
| LOS7      | LOS >7                                                                                          | ‘unitDischargeOffset’ >7*24*60'                                                       |
| Readm     | Count(‘ICUSTAY ID’) >1                                                                          | Count(‘patientUnitStayID’) > 1                                                        |
| Fi_ac     | class('hospitaldischargelocation')    and (Timegap< discharge time -INTIME < prediction window) | class(DISCHARGE_LOCATION) and (Timegap< discharge time -INTIME <   prediction window) |
| Im_disch  | class('hospitaldischargelocation')                                                              | class(DISCHARGE_LOCATION)                                                             |
| Dx        | ICD9 CODE-LONG TITLE (MIMIC-III) ICD10 CODE-LONG TITLE (MIMIC-IV)                               | set(‘diagnosisstring’) per 1 ICU                                                      |

## Implementation details
### Model architecture for each model
- Comparison models on detail.

|   | **Embedding** | **Feature** | **Structure**                                 |
|----------------|---------------|-------------|------------------------------------------------|
| **UniHPF**     | Text based    | Entire      | Hierarchical (Transformer 2 layer + 2 layer )  |
| **DescEmb***   | Text based    | Selected    | Hierarchical (Transformer 2 layer + 2 layer )  |
| **Rajikomar*** | Code based    | Entire      | Hierarchical (Transformer 2 layer + 2 layer )  |
| **SAnD***      | Code based    | Selected    | Flatten (Transformer 4 layer )                 |

- UniHPF and baseline models can be distinguished in the veiw of embedding method, feature usage and model structrue.

### Hyperparameters
- We searched for the ideal set of hyperparameters for each case for more than 72 hours. We found that the hyperparameters had little impact on the outcome. 
- We combined one set of hyperparameters for all cases to make the experiment more straightforward without significantly degrading the performance of each individual model. 
- The final results show a dropout of 0.3, a predictive model's embedding dimension being 128 and a learning rate of 1e-4.
### Computational resources
- VRAM usage of each model and parameters

|                　                |  **SAnD***  | **DescEmb*** | **Rajikomar*** |      **UniHPF**     |
|:--------------------------------:|:-------:|:--------:|:----------:|:---------------:|
|              Memory              |  8.9GB  |  65.1GB  |   35.4GB   | 78.8GB   * 2GPU |
| Total   Parameters               | 1746945 | 4414465  |  1970561   |     4414465     |
| Parameters   w/o embedding layer | 1056897 |  396929  |   396929   |      396929     |

- Memory was recorded when the batch size was 128 based on the LOS3 prediction task, which is a binary classification in single domain training.
- In the case of the flattened model SAnD, the input sequence length is 8192, but the memory is much reduced by using a performer. We will discuss flatten in more detail in the flatten vs hierarchical section below.
### Training details
- We splitted train set, valid set, test set with 9:1:1 ratio and split is stratified for each prediction task.
- Training model is saved for best prediction performance at valid testset and early stopping with 10 epoch patience is applied.
- For pooled learning, a model with pooled datasets is trained and evaluated  for a valid set of each dataset. Test best performance model on each dataset.
- For transfer learning, a single domain trained model with source datasets is loaded and used for zero-shot learning or further fine-tuning on target datasets. 
## Hierarchical vs Flatten model
- Ablation study for hierarchical versus flattened model

|  | **MIMIC-III** |      |       |     | **eICU**   |       |     |    | **MIMIC-IV** |     |       |       |
|------------------|---------------|-----------------|---------------|-------------|------------|-----------------|---------------|-------------|--------------|-----------------|---------------|--------------|
|          | SAnD* (fl)    | Rajikomar* (hi) | DescEmb* (hi) | UniHPF (hi) | SAnD* (fl) | Rajikomar* (hi) | DescEmb* (hi) | UniHPF (hi) | SAnD* (fl)   | Rajikomar* (hi) | DescEmb* (hi) | UniHPF (hi)  |
| **Flatten**      | 0.086         | 0.094           | 0.085         | 0.078       | 0.407      | 0.403           | 0.396         | 0.401       | 0.123        | 0.116           | 0.117         | 0.120        |
| **Hierarchical** | 0.084         | 0.093           | 0.068         | 0.061       | 0.403      | 0.404           | 0.402         | 0.409       | 0.120        | 0.115           | 0.105         | 0.118        |
| **Flatten**      | 0.263         | 0.316           | 0.277         | 0.290       | 0.165      | 0.169           | 0.135         | 0.148       | 0.287        | 0.318           | 0.275         | 0.294        |
| **Hierarchical** | 0.290         | 0.326           | 0.291         | 0.327       | 0.164      | 0.172           | 0.162         | 0.169       | 0.311        | 0.317           | 0.292         | 0.315        |
| **Flatten**      | 0.662         | 0.663           | 0.657         | 0.661       | 0.585      | 0.588           | 0.574         | 0.570       | 0.604        | 0.624           | 0.592         | 0.609        |
| **Hierarchical** | 0.662         | 0.663           | 0.665         | 0.666       | 0.584      | 0.585           | 0.577         | 0.583       | 0.619        | 0.636           | 0.607         | 0.648        |
| **Flatten**      | 0.365         | 0.359           | 0.364         | 0.358       | 0.286      | 0.289           | 0.276         | 0.272       | 0.317        | 0.335           | 0.305         | 0.317        |
| **Hierarchical** | 0.364         | 0.365           | 0.366         | 0.366       | 0.282      | 0.284           | 0.281         | 0.285       | 0.313        | 0.331           | 0.313         | 0.328        |

- For giving the same information between hierarchical and flatten models, We restricted the number of events for each sample. Due to the computation resource limitation, flattened models use 8192 as maximum sequence length and corresponding number of events is used on hierarchical model input.
- Experiments were conducted to compare the structures of each model in flatten and hierarchical cases. The origin structure of each model is displayed in parenthesis. In most cases, hierarchical performance is higher than flatten structure regardless of model type.
- This confirmed that embedding and aggregation of time-series EHR in event units is a more favorable condition for the model.

## Pre-training
- For pre-training, fine-tuning is performed after pre-training with the entire input dataset except for the test set. In the context of conventional pre-training and transfer learning, transfer learning from a large hospital to a small hospital can be considered.
- However, we only checked whether learning from pre-training gives benefits to the model or not compared to random initialization of model parameters, rather than fine-tuning on partial datasets after pre-training on the entire dataset. The experiment on the transfer situation from a large hospital to a small hospital is left as future work.

- In DescEmb with a hierarchical structure, pretraining text within each event with MLM was performed in the event encoder part, but no significant performance improvement was seen. So, we proceeded with pretraining in the structure of flatten where we expect events can be seen by each other, rather than just learning text within the same event. 

- SPAN MLM is intended to learn the context of the EHR time series event by learning the event itself, rather than simply learning the partial random masked subword of the description.
- Pre-training results 

|               | **Pretraining Dataset** |              |  **MIMIC-III + MIMIC-IV +   eICU** |            |        |         |
|---------------|:-------------------:|:------------:|:------------------------------:|:----------:|:------:|:-------:|
|               | **Model**               | **Hierarchical** |                                |   **Flatten**  |        |         |
| **Eval Datasets** | **Task**                | **UniHPF (hi)**   | **Wav2Vec**                        | **UniHPF(fl)** | **MLM**    | **SPANMLM** |
|   **MIMIC-III**   | Mort                | 0.327        | 0.325                          | 0.290      | 0.291  | 0.293   |
|               | LOS3                | 0.666        | 0.663                          | 0.661      | 0.664  | 0.663   |
|               | LOS7                | 0.366        | 0.364                          | 0.358      | 0.358  | 0.357   |
|               | Readm               | 0.061        | 0.601                          | 0.078      | 0.068  | 0.073   |
|               | Fi_ac               | 0.617        | 0.616                          | 0.600      | 0.606  | 0.601   |
|               | Im_disch            | 0.390        | 0.389                          | 0.375      | 0.379  | 0.379   |
|               | Dx                  | 0.759        | 0.761                          | 0.753      | 0.756  | 0.755   |
|      **eICU**     | Mort                | 0.169        | 0.167                          | 0.148      | 0.150  | 0.151   |
|               | LOS3                | 0.583        | 0.579                          | 0.570      | 0.574  | 0.572   |
|               | LOS7                | 0.285        | 0.281                          | 0.272      | 0.278  | 0.278   |
|               | Readm               | 0.409        | 0.404                          | 0.401      | 0.402  | 0.400   |
|               | Fi_ac               | 0.582        | 0.574                          | 0.560      | 0.558  | 0.561   |
|               | Im_disch            | 0.559        | 0.558                          | 0.543      | 0.545  | 0.547   |
|               | Dx                  | 0.689        | 0.685                          | 0.656      | 0.657  | 0.660   |
|    **MIMIC-IV**   | Mort                | 0.315        | 0.307                          | 0.294      | 0.296  | 0.294   |
|               | LOS3                | 0.648        | 0.644                          | 0.609      | 0.613  | 0.614   |
|               | LOS7                | 0.328        | 0.323                          | 0.317      | 0.315  | 0.316   |
|               | Readm               | 0.118        | 0.112                          | 0.120      | 0.119  | 0.120   |
|               | Fi_ac               | 0.724        | 0.722                          | 0.714      | 0.717  | 0.719   |
|               | Im_disch            | 0.412        | 0.410                          | 0.368      | 0.373  | 0.372   |
|               | Dx                  | 0.834        | 0.836                          | 0.816      | 0.817  | 0.817   |

- The MLM accuracy of random masking is more than 90%, but the accuracy of span MLM is about 80%, resulting in a more difficult task for the model.
- We haven’t seen any performance improvement with pre-training yet. A pre-training method suitable for the characteristics of EHR is needed to be newly developed.

## Qualitative analysis
- Next, to test if UniHPF can handle the discrepancy between MIMIC-III and eICU in terms of textual description, we select four drug terms from the top 15 features Table~ that exist in both datasets, and swap a part of terms between the two datasets, where the selected terms are described in Table ~}.
- For example, we switch all existing drugs ``vancomycin hcl'' in the test set of MIMIC-III to ``vancomycin in ivpb''.
- Then, we evaluate our model that was trained on each single dataset for mortality prediction, using the modified test set of MIMIC-III and eICU, respectively.

- Terms used for qualitative analysis

| **Term**                | **MIMIC-III**               | **eICU**                        |
|---------------------|-------------------------|-----------------------------|
|   vancomycin        |  hcl                    |   in ivpb                   |
|     morphine        |  sulfate oral soln. po  |  250 mg sodium chloride     |
|     norepinephrine  |  -                      |  bitartrate iv              |
|     acetaminophen   |  iv                     |  oxycodone 325 mg po   tabs |

- As a result, the AUPRC decreased marginally (0.8%p and 0.6%p in MIMIC-III and eICU, respectively) although the model never saw the modified features before (e.g., "vancomycin" in ivpb if the model has been trained on MIMIC-III). 
- We conclude that UniHPF is able to deal with distinct EHR datasets, as long as they are based on the same language.

# Getting started with UniHPF
## STEP 1 : Installation
Requirements

* [PyTorch](http://pytorch.org/) version >= 1.9.1
* Python version >= 3.8

## STEP 2: Prepare training data
First, download the dataset from these links: 
	[MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
[MIMIC-IV](https://physionet.org/content/mimiciv/2.0/)
[eICU](https://physionet.org/content/eicu-crd/2.0/)
[ccs_multi_dx_tool_2015](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip)
[icd10cmtoicd9gem](https://data.nber.org/gem/icd10cmtoicd9gem.csv)
	
Note that you will need to get access for each dataset, publicly available electronic health records. 

Second, for preparation your data, make directory structure like below:
```
data_input_path
├─ mimic3
│  ├─ ADMISSIONS.csv
│  ├─ PATIENTS.csv
│  ├─ ICUSYAYS.csv
│  ├─ LABEVENTES.csv
│  ├─ PRESCRIPTIONS.csv
│  ├─ PROCEDURES.csv
│  ├─ INPUTEVENTS_CV.csv
│  ├─ INPUTEVENTS_MV.csv
│  ├─ D_ITEMDS.csv
│  ├─ D_ICD_PROCEDURES.csv
│  └─ D_LABITEMS.csv
├─ eicu
│  ├─ diagnosis.csv
│  ├─ infusionDrug.csv
│  ├─ lab.csv
│  ├─ medication.csv
│  └─ patient.csv
├─ mimci4
│  ├─ admissions.csv
│  ├─ …
│  └─d_labitems.csv
├─ ccs_multi_dx_tool_2015.csv
└─ icd10cmtoicd9gem.csv

```
```
data_output_path
├─single
├─transfer
├─pooled
├─label
└─fold
```
Then run preprocessing code
```shell script
$ python preprocess_main.py 
    --data_input_path $csv_directory
    --data_output_path $run_ready_directory 
```
Note that pre-processing takes about 6hours in 128 cores of AMD EPYC 7502 32-Core Processor, and requires 180GB of RAM.


STEP 3. Training a new model
Other configurations will set to be default, which were used in the UniHPF paper.
$data should be set to 'mimic3' or 'eicu' or ‘mimic4’ 
`$model` should be set to one of [‘SAnD’, ‘Rajkomar’, ‘DescEmb’, ‘UniHPF’, ‘Benchmark’]

`$task` should be set to one of ['readm’, ‘los3', ‘los7’, ‘mort’, ‘fi_ac’, ‘im_disch’, ‘dx’]

Note that `--input-path ` should be the root directory containing preprocessed data.
### Example
### Train a new UniHPF model:

```shell script
$ python main.py \
    --input_path /path/to/data \
    --model UniHPF \
    --train_type single  \
    --src_data mimic3 \
    --train_task predict  \
    --pred_task $pred_task \
    --batch_size $batch_size \
    --device_num $device_num \
```
Note: if you want to train with pre-trained model, add command line parameters `--load_pretrained_model` and add directory of pre-trained model checkpoint

### Pre-train UniHPF model:

```shell script
$ python main.py \
    --input_path /path/to/data \
    --model UniHPF \
    --src_data $data \
    --train_task pretrain  \
    --pretrain_task $pretrain_task \
    --batch_size $batch_size \
    --device_num $device_num \
```

## Pooled learning 
```shell script
$ python main.py \
    --input_path /path/to/data \
    --model UniHPF \
    --train_type pooled  \
    --src_data mimic3_eicu \
    --train_task predict  \
    --pred_task $pred_task \
    --batch_size $batch_size \
    --device_num $device_num \
```


## Transfer learning
```shell script
$ python main.py \
    --input_path /path/to/data \
    --model UniHPF \
    --train_type transfer  \
    –ratio 0 \
    --src_data mimic3 \
    --target_data eicu \
    --train_task predict  \
    --pred_task $pred_task \
    --batch_size $batch_size \
    --device_num $device_num \
```

Note that `--ratio` indicates proportion of target dataset for few-shot learning settings. (if ratio is set to zero, then it is zero shot learning) 


# License
This repository is MIT-lincensed.

<!-- # Citation
Please cite as:
```
@misc{hur2022unihpf,
      title={UniHPF: Universal Healthcare Predictive Framework with Zero Domain Knowledge}, 
      author={Kyunghoon Hur and Jungwoo Oh and Junu Kim and Min Jae Lee and Eunbyeol Choi and Jiyoun Kim and Seong-Eun Moon and Young-Hak Kim and Edward Choi},
      year={2022},
      eprint={2207.09858},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
``` -->
