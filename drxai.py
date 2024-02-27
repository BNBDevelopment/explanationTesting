from doctorailib import doctorai
from doctorXAIlib import doctorXAI
import pickle
import numpy as np
import pandas as pd

print("Hello")
import warnings
warnings.filterwarnings('ignore')
from doctorailib import pre_processing

# qui ci va il path dove hai salvato le tabelle ADMISSIONS.csv and DIAGNOSES_ICD.csv di MIMIC-III (già unzippate)
mimic_path = 'D:\\research\\craven\\baselines\\explanationTesting\\data\\_mimicCSVs\\mimic-iii-clinical-database-1.4\\'
# qui ci va il path dove vuoi che ti salvi l'output del preprocessing:
output_path = './preprocessing_doctorai/'
# qui ci va il path dove si trova il csv del CCS grouper
CCS_grouper_csv_file_path = 'D:\\research\\craven\\baselines\\DrXAI\\doctorXAI_working_example\\doctorailib\\'
# CCS = True perchè vogliamo che doctorAI predica i codici CCS e non gli ICD9
# i codici CCS sono dei codici che raggruppano insieme i codici ICD9 in gruppi clinicamente sensati.
pre_processing.prepare_mimic(mimic_path=mimic_path, CCS_grouper_csv_file_path=CCS_grouper_csv_file_path, output_path=output_path, CCS=True)

import datetime
today = datetime.datetime.today()

seqFile="./preprocessing_doctorai/visit"
labelFile="./preprocessing_doctorai/label"
outFile=f"../models/trained_doctorAI_output/{today.year}_{today.month}_{today.day}_MIMIC_III_"

dr = doctorai.DoctorAI(ICD9_to_int_dict="./preprocessing_doctorai/ICD9_to_int_dict",
                       CCS_to_int_dict="./preprocessing_doctorai/CCS_to_int_dict",
                       verbose=True)

dr.train(seqFile=seqFile,
                  inputDimSize=4880,
                  labelFile=labelFile,
                  numClass=272,
                  outFile=outFile,
                  max_epochs=50)


#visualize the path containing the trained model path:
dr.modelFile