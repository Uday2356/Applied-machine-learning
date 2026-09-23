    The decision tree is Model algorithm used for the classification it include all concept same as tree 
    1 = branch
    2 = root
    3 = Leaf

    How to choose Root node : 

    WE use entrophy and information Gain to choose Root node


    ** Process for Decision tree

    1 = start with data 
    2 = choose the best features
    3 = make branches
    4 = repeat the process 
    5 STOP :
       -> Maximum depth get reached
       -> all data pure you get leaf node


when the data is contunues in the column we use MSE for information gain



#**********************SUPPORT VECTOR MACHINE(svm)**************#


MARGIN = The space between the line and nearest point
HyperPlane = A line that separate two categories
 WE TRY TO GET MAXIMUM MARGIN BETWEEN CLASSES 

 IN 2D we call HYPERLINE
 IN 3D we call HYPERPLANE

 
THE DATA POINTS OF THE CLASS THAt are closest THE  MARGINAL  ARE CALLED SUPPORT VESTORS.
 thye support the line 

 #kERNEL Function :

 ## We use the Kernel Method (Kernel Trick) in SVM when the data is not linearly separable in the original feature space.

 when we have data as  
        ○ ○ ○
     ○   ×   ○
    ○  × × ×  ○
     ○   ×   ○
       ○ ○ ○  HERE THE X , O are difficult to separate so we use kernel Function 

       we use kernel function as it moves element from  current dimension to a higher dimension where they become linerly separable because when one type moves up and another at lower we can draw a hyperplane between them.

       