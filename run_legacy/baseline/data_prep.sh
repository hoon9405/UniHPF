rawdata_path=$1
inputdata_path=$2

# python ../preprocess/dataset_construct.py --rawdata_path $rawdata_path --inputdata_path $inputdata_path ;
# python ../preprocess/main_step1.py --rawdata_path $rawdata_path --inputdata_path $inputdata_path ;
python ../preprocess/main_step2.py --rawdata_path $rawdata_path --inputdata_path $inputdata_path ;
# python ../preprocess/npy_script.py --rawdata_path $rawdata_path --inputdata_path $inputdata_path ;

