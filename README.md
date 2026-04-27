### Numerical-Scientific-Computing
Course Projects

Rename the file to change to ssh file: Rename-Item -Path "num_sci_com.txt" -NewName "num_sci_com"


### Config File for Numerical Scientific Computing
```
Host numlab-1
    HostName 10.92.1.34
    User ubuntu
    IdentityFile ~/.ssh/num_sci_com
    IdentitiesOnly yes
```

### Connect to the virtual environment 
source /home/ubuntu/Numerical-Scientific-Computing/num_com/bin/activate 

### To activate github
ssh -T git@github.com