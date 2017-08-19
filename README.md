#HebNet
HebNet - Deep Neural Network for predicting hebrew characters and digits
 
Authors: Maor Sisso and Shmuel Amar

#Installation
to install hebnet run:

```
make setup
```

#Example Usage
get help about the optional parameters:

```
export PYTHONPATH=".:$PYTHONPATH"
python hebnet --help
```


for classifying a directory:

```
python hebnet --images-dir path/to/images/dir/ 
```

for classifying a single file:

```
python hebnet --images-dir path/to/images/dir/file.jpg 
```
