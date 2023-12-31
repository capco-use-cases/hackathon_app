Instructions:
1. Running environment is Google Colab
2.	Create a directory called ‘project’ anywhere inside ‘/content/drive/MyDrive/’ of google colab window.
3.	Upload all the files inside ‘project’ zipped directory from the repository to the 'project' directory created in the above step.
4.	Open the ‘frontend.ipynb’ file and execute cell by cell.
5.	Please make sure to assign the parent path variable in the 'frontend.ipynb' file properly with the correct path of the ‘project’ directory path created.
6.	Double click on the cell containing the form in 'frontend.ipynb' file if required to maximize the form and hide the backend code for better aesthetics.
7.	Fill in the fields in the form and execute it to get output. And other required instructions are mentioned in the ipynb file itself.

<br><br>

Notes:
1.	‘record.csv’ file is performing the role of a database in this use case, which contains some already tested instances.
2.	After a query is performed in the app, if the customer instance is absent in the record.csv database, the prediction and recommendation is done and is updated in the database. Whereas, if any instance of the customer is already present, the last prediction already made is spilled and confirmation is asked if the user wants to add another instance in the record as shown below:
   
![image](https://github.com/capco-use-cases/hackathon_app/assets/141928608/e00ccee9-1ff9-4509-a8bb-82d0845ed697)

3. Contents of ‘Overview.pdf’ file attached:
    -	A. Libraries used for the project
    -	B. Preprocessing steps
    -	C. Training steps and discussions
    -	D. A few points on EDA
    -	E. Notes regarding approach

<br><br>

Frontend app snapshot:
![image](https://github.com/capco-use-cases/hackathon_app/assets/141928608/a9d2dd9e-999f-4e14-8f9c-c87d3de4f721)
