# comp9417_ass2
COMP 9417 Assignment 2

To install packages:
1. Check that venv is installed: 
	Windows: py -m pip install --upgrade pip
		 py -m pip install --user virtualenv
	Linux and macOS: python3 -m pip install --user --upgrade pip
			 pythom3 -m pip install --user virtualenv

2. Activate the venv:
	Windows: .\pjpk\Scripts\activate
	Linux and macOS: source pjpk/bin/activate

3. Install the packages: pip install -r requirements.txt
4. To install spacy models:
 	Windows: py -m spacy download en-core-web-lg
		 py -m spacy download en-core-web-sm
	Linux and macOS: py -m spacy download en-core-web-lg
			 py -m spacy download en-core-web-sm
