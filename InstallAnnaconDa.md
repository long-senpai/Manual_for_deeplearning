# How to install the Annaconda 
## Anaconda is uses to manage the Environment to play with Deeplearning

** Requirement **
* Ubuntu OS Installed (vers >= 18.04)

** STEP 1 **

Download anaconda from the wbesite

``` 
	$ wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
```
Then
`   $ chmod +x Anaconda3-2020.11-Linux-x86_64.sh `

** STEP 2 **
Install extend dependcies for QT

`   $ sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 `

** STEP 3  **

Installation

`   $ bash ~/Downloads/Anaconda3-2020.11-Linux-x86_64.sh ` 
Just press enter or "yes" from the terminal. 

Add anaconda to the Bashrc file

** STEP 4 **

```
   $ gedit ~/.bashrc
```
Then add follow lines into the bottom of basrhc:

> 
	export PATH=~/anaconda3/bin:$PATH 
Defaull the ananconda will install in the /home/user folder
Then save and close.

Now reload bashrc by command: 
`   $ source ~/.bashrc `
For ensure, you should close the terminal and create a new termial session

** STEP 5 **
 Create a new env with Anaconda

`   $ conda create -n name_of_env python=3.8 ` 

So now we had successful init a env with anaconda. To activate the env.

`   $ conda activate name_of_env ` 
Each time you open a new termail, you have to activate the env again. 






