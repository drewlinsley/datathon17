# Logging into Brown CIS stronghold to analyze Neurodatathon data.
### Because the dataset used in this competition is so special, it is being securely held in Brown's stronghold computing environment. This means that in order to analyze the data you must (1) have stronghold login credentials from CIS, and (2) write and run your analysis code exclusively within the stronghold.

1. If you have signed up for the Neurodatathon, you will receive an email from CIS with your login information.
	- To access stronghold you will need the "Microsoft Remote Desktop" (MRD) app. This should be built into Windows machines. It can be downloaded in Mac and Unix appstores/package managers.
	- Open the MRD app and select "new". Enter the following information (fill in your information where there are quotations):
		+ Connection name: datathon
		+ PC name: datathon.stronghold.brown.edu
		+ User name: ad\"your brown username"
		+ Password: "your brown password"
	- After saving your connection preferences double-click on the session. You likely will be asked to first re-enter your password, then prompted for Duo two-factor authentication. Once you authenticate, you will be transfered to a windows desktop. (Contact Neurodatathon organizers if you run into issues).
	- Inside of the windows desktop, double click "MobaXterm", which will give you a graphical interface to the Stronghold computing environment.
		+ Click SSH
		+ Under Basic SSH settings:
			o Remote host: 192.168.156.100:22
			o Click the checkbox next to "Specify username"
			o For username, type the username given to you by CIS
			o The first time you log in you will be prompted to change your password. It must be dissimilar from the original password supplied by CIS.
		+ If you see a text/terminal interface that says `-bash-4.2$` you're all set!
2. Analyze Neurodatathon data within the Stronghold environment.
	- In the MobaXterm terminal window, type `jupyter notebook`.
	- This will load a web browser that lists the contents of your current directory.
	- Click on datathon17. You will see a number of files: optimize_*.py and create_*.py files that can serve as templates for your analyses; a c_utils.py file with competition utility functions that you will probably not need to worry about; a folder with data in it; and the competition README.md file.
	- Click on `optimize_sklearn_decoding.py`. This will load a tab showing the script. Locate the tab you were on a moment ago and click on it (it should say `datathon17/`).
	- On the right side of the screen there is a dropdown menu that says `New`. Click this, then click `Python 2 notebook`.
	- You now have a python interpreter session open! Flip back to the `optimize_sklearn_decoding.py` tab, and select and copy all of the text (first ctrl + a, then ctrl + c).
	- Flip back to the tab with the Python notebook you just opened (`Untitled`). Paste the text you just copied (ctrl + v or right-click and paste).
	- Enter your name or team-name in the field in the script that says "team_name". Now click run on the top of the screen (looks like a play button).
	- Congratulations! You have built your first brain machine interface to decode behavior from neural activity! You should see the following output:
	```
	0 X Validation correlation is 0.396328475383
	1 Y Validation correlation is 0.296628687071
	2 X Validation correlation is 0.372368854327
	3 Y Validation correlation is 0.300677922575
	4 X Validation correlation is 0.353888233982
	5 Y Validation correlation is 0.284602376885
	6 X Validation correlation is 0.366446206758
	7 Y Validation correlation is 0.276715919945
	8 X Validation correlation is 0.359704989079
	9 Y Validation correlation is 0.29312340295
	10 X Validation correlation is 0.381112253429
	11 Y Validation correlation is 0.294684570941
	12 X Validation correlation is 0.356302131907
	13 Y Validation correlation is 0.317856828035
	14 X Validation correlation is 0.395453750638
	15 Y Validation correlation is 0.291813000881
	16 X Validation correlation is 0.36294784109
	17 Y Validation correlation is 0.309849167776
	18 X Validation correlation is 0.397903644685
	19 Y Validation correlation is 0.29213351064
	20 X Validation correlation is 0.362538400582
	21 Y Validation correlation is 0.298783085091
	22 X Validation correlation is 0.399270289404
	23 Y Validation correlation is 0.289222977244
	24 X Validation correlation is 0.362492624076
	25 Y Validation correlation is 0.294430894931
	26 X Validation correlation is 0.361230346528
	27 Y Validation correlation is 0.275249367956
	28 X Validation correlation is 0.354696104389
	29 Y Validation correlation is 0.271406890081
	30 X Validation correlation is 0.347548686386
	31 Y Validation correlation is 0.327194927711
	32 X Validation correlation is 0.373294598358
	33 Y Validation correlation is 0.293946591301
	34 X Validation correlation is 0.344067086489
	35 Y Validation correlation is 0.313850469906
	36 X Validation correlation is 0.375215347916
	37 Y Validation correlation is 0.250300930281
	38 X Validation correlation is 0.344738629436
	39 Y Validation correlation is 0.325472184496
	40 X Validation correlation is 0.381347327582
	41 Y Validation correlation is 0.244408399441
	42 X Validation correlation is 0.344615293525
	43 Y Validation correlation is 0.312397328895
	44 X Validation correlation is 0.383554641882
	45 Y Validation correlation is 0.248508395194
	```
	- You should plan to use the "optimize_" or "create_" scripts as templates for your analyses. They handle all of the data loading and saving, and you swap in and out computational mechanisms (from keras or Scikit-learn) to build your analysis.
3. You can close your connection to the stronghold at any time by simply closing the Microsoft Remote Desktop app. If you want a more graceful exit, click on the MobaXterm window and press `ctrl + c`. This will give you the option to save and close your Python notebook. After this is complete you can exit the connection by closing the Microsoft Remote Desktop app.

4. In case you run into an issue that the competition organizers cannot resolve, you will be instructed to send an email to:
	To: stronghold-help@brown.edu
	Subject: Datathon
	Body: Describe your problem and include your username. Also indicate the jumpbox you connected to and the name of your VM.
