# UniHPF : Universal Healthcare Predictive Framework with Zero Domain Knowledge

UniHPF is a universal healthcare predictive framework, which requires no medical domain knowledge and minimal preprocessing for multiple prediction tasks. Given any time-series EHR, 		 	 	 			
					
Our framework presents a method for embedding any form of EHR systems for prediction tasks with- out requiring domain-knowledge-based pre-processing, such as medical code mapping and feature selection.  
				
This repository provides official Pytorch code to implement UniHPF, a universal healthcare predictive framework.

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
