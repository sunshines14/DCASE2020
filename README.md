# DCASE2020
This repository is the official implementations of our DCASE 2020 task 1a with technical report:

Soonshin Seo, Changmin Kim and Ji-Hwan Kim: "MULTI-CHANNEL FEATURE USING INTER-CLASS AND INTER-DEVICE STANDARD DEVIATIONS FOR ACOUSTIC SCENE CLASSIFICATION ", submitted to task 1a of the 2020 DCASE Challenge 

A technical report link at http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Kim_87.pdf
  
## Training
 1. Download the data form links at http://dcase.community/challenge2020/task-acoustic-scene-classification#external-data-resources
 2. Run the script "train.py" (for 2-path residaul CNN) or "train_4path.py" (for 4-path residual CNN)
	  
## Evauation
 1. Run the script "evaluate.py"
		 
## Acknowledgement
We used the implementation presented in https://github.com/McDonnell-Lab/DCASE2019-Task1 as our baseline script.

## Bibtex
@techreport{Soonshin2020,
    Author = "Soonshin Seo, Changmin Kim and Ji-Hwan Kim",
    title = "Multi-Channel Feature Using Inter-Class and Inter-Device Standard Deviations for Acoustic Scene Classification",
    institution = "DCASE2020 Challenge",
    year = "2020"
}
