import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#########################
#
# Data
#
#########################

def gray_image(img_arr):
    """
    Function to greyscale a given image
    """
    gray_img_arr = np.zeros_like(img_arr, order="F") #Create empty array for greyscaled image in order "F" (It helps to not lose data when the matrix is flattened into an array)
    for row in range(img_arr.shape[0]):
        for pixel in range(img_arr.shape[1]):
            gray_img_arr[row][pixel] = (0.2126*img_arr[row][pixel][0] + 0.7152*img_arr[row][pixel][1] + 0.0722*img_arr[row][pixel][2]) #Calculate relative luminesance
            gray_img_arr[row][pixel][3] = 1 #Makes sure the pixel is not transparent if image needs to be saved
    return gray_img_arr

#Load images
bangladesh_flag = mpimg.imread("flag/Bangladesh.png")
denmark_flag = mpimg.imread("flag/Denmark.png")
georgia_flag = mpimg.imread("flag/Georgia.png")
japan_flag = mpimg.imread("flag/Japan.png")
switzerland_flag = mpimg.imread("flag/Switzerland.png")
sweden_flag = mpimg.imread("flag/Sweden.png")
norway_flag = mpimg.imread("flag/flag-of-Norway.png")
guinea_flag = mpimg.imread("flag/flag-of-Guinea.png")
finland_flag = mpimg.imread("flag/flag-of-Finland.png")
france_flag = mpimg.imread("flag/flag-of-France.png")
ireland_flag = mpimg.imread("flag/flag-of-Ireland.png")
italy_flag = mpimg.imread("flag/flag-of-Italy.png")
england_flag = mpimg.imread("flag/England.png")

#Greyscale images
DK = gray_image(denmark_flag)
JP = gray_image(japan_flag)
BL = gray_image(bangladesh_flag)
GG = gray_image(georgia_flag)
SL = gray_image(switzerland_flag)
SW = gray_image(sweden_flag)
NW = gray_image(norway_flag)
GI = gray_image(guinea_flag)
FL = gray_image(finland_flag)
FR = gray_image(france_flag)
IL = gray_image(ireland_flag)
IY = gray_image(italy_flag)
EN = gray_image(england_flag)

#Convert greyscaled matrices into 1D arrays
DK_vec = denmark_flag.flatten()#DK.flatten()
#JP_vec = japan_flag.flatten()#JP.flatten()
#BL_vec = bangladesh_flag.flatten()#BL.flatten()
#GG_vec = georgia_flag.flatten()#GG.flatten()
SL_vec = switzerland_flag.flatten()#SL.flatten()
SW_vec = sweden_flag.flatten()#SW.flatten()
NW_vec = norway_flag.flatten()#NW.flatten()
GI_vec = guinea_flag.flatten()#GI.flatten()
FL_vec = finland_flag.flatten()#FL.flatten()
FR_vec = france_flag.flatten()#FR.flatten()
IL_vec = ireland_flag.flatten()#IL.flatten()
IY_vec = italy_flag.flatten()#IY.flatten()
EN_vec = england_flag.flatten()#EN.flatten()

def test_flag(vec): #This function adds a smal value to a given flag and returns it, this is to see if the PCA works as predicted (This output should lie close to the input on the score plot)
    return vec+0.1

data = pd.DataFrame({'DK':DK_vec,
                     #'Japan':JP_vec,
                     #'Bangladesh':BL_vec,
                     #'Georgia':GG_vec,
                     #'Switzerland':SL_vec,
                     #'Test':test_flag(SL_vec),
                     "SE":SW_vec,
                     "NO":NW_vec,
                     #"Guinea":GI_vec,
                     "FI":FL_vec,
                     "FR":FR_vec,
                     "IE":IL_vec,
                     "IT": IY_vec,
                     "GB":EN_vec
                     }).T #Dataframe for each of the flags

#########################
#
# Perform PCA on the data
#
#########################

normalized_data = data - data.mean()
 
U, S, Vt = np.linalg.svd(normalized_data, full_matrices=False) #Reduced SVD

PCs = np.matmul(U, np.diag(S)) #Matrix-matrix product to find principal components

#List of principal components is created
PC_list = []
for i in S**2:
    PC_list.append(round((i/np.sum(S**2))*100, 2))

labels = [str(i) for i in range(1, len(PC_list)+1)] #Labels PC1, PC2 ... PCn, where n is the number of flags, are created
PCA_df = pd.DataFrame(PCs, index=data.index, columns=labels) #A pandas dataframe where each column is named PC1, PC2, etc. and flags names are used as indexes, is created
#print(PCA_df)

#Add flags to compare to the flags in dataset
#PCA_df.loc["IE"] = np.matmul((IL_vec - data.mean()), Vt.T)
#PCA_df.loc["GB"] = np.matmul((EN_vec - data.mean()), Vt.T)

#gamle SCREE plots
"""#Create scree plot
f, ax = plt.subplots(1, 1)
ax.bar(range(len(PC_list)), PC_list, tick_label=labels)
ax.xaxis.set_tick_params(labelsize=14) #Font-size of xaxis
ax.yaxis.set_tick_params(labelsize=14)
ax.set_ylabel("Percentage of Explained Variance", size=14)
ax.set_xlabel("Principal Component", size=14)
ax.set_title("Scree plot", size=16)
#plt.savefig("Scree plot flag.png", dpi=500)
plt.show() #Shows plot then resets it"""


#NYE (Martin approved) scree plots
f, ax = plt.subplots(1, 1) # A figure with 1 plot
ax.plot(range(1, len(PC_list)+1), PC_list, '-', color='brown', zorder=1) # Plot the line with zorder 1
ax.scatter(range(1, len(PC_list)+1), PC_list, facecolors='white', edgecolors='brown', zorder=2) # Use scatter plot with white-filled circles and zorder 2
ax.xaxis.set_tick_params(labelsize=14) # Font-size of tick_labels on the xaxis
ax.yaxis.set_tick_params(labelsize=14)
ax.set_ylabel("Percentage of Explained Variance", size=14)
ax.set_xlabel("Principal Component", size=14)
ax.set_title("Scree plot", size=16)
#plt.savefig("Scree plot flag.png", dpi=500)
plt.show() # Shows plot then resets it

#Create score plot
f, ax = plt.subplots(1, 1)
for flag in range(0, len(PCA_df.index), 1): #1 point for each flag with own its label annotated
    ax.scatter(PCA_df.iloc[flag][0], PCA_df.iloc[flag][1], c="#4d994d")
    ax.annotate(PCA_df.index[flag], (PCA_df.iloc[flag][0], PCA_df.iloc[flag][1]), size=14, color="#BB0000") #Each point gets its title annotated on top of it
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.set_xlabel(f"PC1 - {PC_list[0]} %", size=14)
ax.set_ylabel(f"PC2 - {PC_list[1]} %", size=14)
plt.grid()
plt.axhline(y=0, color="k")
plt.axvline(x=0, color="k")
plt.title("Score plot", size=16)
#plt.savefig("Score plot flag.png", dpi=500)
plt.show() #Shows plot then resets it