Within each folder there is a csv generator and reader \n
The generator needs the root of source folder with this format: SourceFolder/Subject/XXX.acq or SourceFolder/Subject/HRVfolder/XXX.puls \n
It also needs a dicom file to align the biosignal to fMRI time length
The csv generator creates csv files for a single run
The csv reader python files are a compilation of function
To do a reading, write the function, input the run.csv and other needed variables to run
