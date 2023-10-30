Instructions:
1. Running environment is Google Colab
2.	Create a directory called ‘project’ anywhere inside ‘/content/drive/MyDrive/’ of google colab window.
3.	Upload all the files inside ‘project’ directory from the repository to the 'project' directory created in the above step.
4.	Open the ‘frontend.ipynb’ file and execute cell by cell.
5.	Please make sure to assign the parent path variable in the 'frontend.ipynb' file properly with the correct path of the ‘project’ directory path created.
6.	Fill in the fields in the form and execute it to get output. And other required instructions are mentioned in the ipynb file itself.

Notes:
1.	‘record.csv’ file is performing the role of a database in this use case, which contains some already tested instances.
2.	After a query is performed in the app, if the customer instance is absent in the record.csv database, the prediction and recommendation is done and is updated in the database. Whereas, if any instance of the customer is already present, the last prediction already made is spilled and confirmation is asked if the user wants to add another instance in the record as shown below:
   
![image](https://github.com/capco-use-cases/hackathon_app/assets/141928608/e00ccee9-1ff9-4509-a8bb-82d0845ed697)

3. Contents of ‘Overview.pdf’ file attached:
    a.	Libraries used for the project
    b.	Preprocessing steps
    c.	Training steps and discussions
    d.	A few points on EDA
    e.	Notes regarding approach
