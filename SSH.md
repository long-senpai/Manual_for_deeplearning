# SSH to the EC2 instance with the key.pem
```
$ chmod 400 nobi.pem
$ ssh -i "nobi.pem" ubuntu@ec2-52-29-53-202.eu-central-1.compute.amazonaws.com
```
# Scp copy/push file from server to local machine
```
$ scp -i "nobi.pem" ubuntu@ec2-52-29-53-202.eu-central-1.compute.amazonaws.com:/home/user_name/folder.zip "destination_path_on_your_local_machine"
Ex: 
$ scp -i "nobi.pem" ubuntu@ec2-52-29-53-202.eu-central-1.compute.amazonaws.com:/home/user_name/folder.zip . -> For the current folder
$ scp -i "nobi.pem" ubuntu@ec2-52-29-53-202.eu-central-1.compute.amazonaws.com:/home/user_name/folder.zip /home/...
----------------------------------------------------------------------------------------------------------------------
$ scp /path/to/your/file/ -i "nobi.pem" ubuntu@ec2-52-29-53-202.eu-central-1.compute.amazonaws.com:/home/user_name/folder_destination
