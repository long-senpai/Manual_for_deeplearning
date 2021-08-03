# INSTALL DOCKER ON UBUNTU 20.04
	
**STEP1**
##### Use the apt command to install the docker.io package:

```python3
	$ sudo apt install docker.io
```
**STEP2**
##### Start docker and enable it to start after the system reboot:
```python
	$ sudo systemctl enable --now docker
```
**STEP3**
##### Optionally give any user administrative privileges to docker:
```python
	$ sudo usermod -aG docker ${USER} 
```
**STEP4** 
##### Check docker version:
```python
	$ docker --version
```
# DOCKER EXAMPLEs:
#### 1: Create a docker image, build -> run a python file
* Item name dockerfile:
```docker
	FROM python:3

	# define the working space contain the example
	WORKDIR /home/long/Desktop/manual/docker

	# copy all source and the depend file into the image
	COPY . .

	# deploy and run
	CMD ["python","./test.py"]
```
* Item name test.py:
```python
	print("hello long-senpai")
```
* Build docker image:
```python
	$ docker build -t testdocker .
```
* The dot on above command indicate for the current folder which contain the docker and source. 
* To check the docker images
```
 $ docker images
```
* then a docker image named testdocker was build.
* To run docker image 
```
 $ docker run  --name example1 testdocker
```
#### 2: Create a docker image uses flaskapp backend
* Item dockerfile:
```docker
	FROM python:3

	# define the working space contain the example
	WORKDIR /home/long/Desktop/manual/docker

	# copy all source and the depend file into the image
	COPY . .

	RUN pip install flask
	
	# deploy and run
	CMD ["python","./test2.py"]
```
* Item test2.py:
```python3
	from flask import Flask

	app = Flask(__name__)

	@app.route('/')
	def test():
	    return "hellu may che"

	if __name__=="__main__":
	    app.run(debug=True , host='0.0.0.0', port='6464', use_reloader=False)
```
* Build 
```language
$ docker build -t example2 . 
```