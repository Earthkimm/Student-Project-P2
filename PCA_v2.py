import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
import scipy.signal as signal

class Fourier():
    """
    Classes are objects that gather related elements together (why it's called object-oriented-programming (OOP)).
    Functions in classes are called "Methods" and should have "self" as first argument to access the objects' own functions and variables.
    A class can start with the "__init__"-function to do some initialising tasks (like defining variables).
    """
    fugl_dataframe = {} #Defining this variable here makes it global for all instances of this object
    def __init__(self, dire, fugl, dataset):
        self.start_Freq_index = 0 #0 #captures the matrix without moidfications
        self.end_Freq_index = 257 #257 #captures the matrix without moidfications
        self.fugl = fugl #The argument "fugl" is saved in the object to be accesed later
        self.s_rate, self.signal = wav.read(dire) #The file in the directory from the "dire"-argument is read and sampling_rate aswel as the signal is saved in the object (Every object therefore gets its own values)
        self.f, self.t, self.Zxx = signal.stft(self.signal, fs=self.s_rate, nperseg=512, noverlap=384, boundary=None, window="hann") #hopsize = 512-384 = 128, window = hann, length of window = 512
        self.Norm = abs(self.Zxx)/abs(np.amax(self.Zxx)) #The stft matrix containing complex values are normalised using their magnitude
        self.Zxx_dB = 20*np.log10(self.Norm/self.Norm.mean()) #Normalised values are converted to dB
        self.Zxx_dB_mod = self.Zxx_dB[self.start_Freq_index:self.end_Freq_index]
        self.Zxx_dB_flat = self.Zxx_dB_mod.flatten() #Matrix to vector
        if dataset: #Data gets saved in the "global" variable if wanted
            self.fugl_dataframe[self.fugl] = self.Zxx_dB_flat

    def pandaF(self): #This method converts the "global" variable "fugl_dataframe" into a pandas dataframe
        self.fugl_pandaframe = pd.DataFrame(self.fugl_dataframe)

    def spect(self): #This method can be called to create a spectrogram of a given bird-call
        plt.pcolormesh(self.t, self.f[self.start_Freq_index:self.end_Freq_index], self.Zxx_dB_mod, shading='gouraud', cmap="Greys")
        #plt.title('STFT Magnitude')
        #plt.ylabel('Frequency [Hz]')
        #plt.xlabel('Time [sec]')
        #plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        #plt.savefig(f"Martins_spectrogrammer/{self.fugl} spectrogram.png", dpi=50)
        plt.show() #Change clf to show if an image is wanted while code is running

#------------------------------------------------------------------------------#
#########################
#
# Data
#
#########################

#Create objects for each bird
"""
Because of the construction of the class Fourier's these rules needs to be followed:
    1. input is the directory of the bird-call
    2. input is the birds name
    3. input is True/False if it is part of the dataset or just converted into the calculated PC-axis
If a bird is not in the dataset it needs to be added in the "new_PCs"-list
"""

#Crows
CROW_1_1 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/CROW_1.1_0.6.wav", "Crow_1", True)
CROW_1_2 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/CROW_1.2_0.6.wav", "Crow_1.2", True)
CROW_1_3 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/CROW_1.3_0.6.wav", "Crow_1.3", True)
CROW_4 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/CROW_4_0.6.wav", "Crow_4", True)
CROW_7 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/CROW_7_0.6.wav", "Crow_7", True)
CROW_8 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/CROW_8.1_0.6.wav", "test_Crow_8", False)

#Gulls
GULL_1 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/GULL_1_0.6.wav", "Gull_1", True)
GULL_2 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/GULL_2_0.6.wav", "Gull_2", True)
GULL_3 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/GULL_3_0.6_2.wav", "Gull_3", True)
GULL_4 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/GULL_4_0.6_2.wav", "Gull_4", True)
GULL_5 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/GULL_5_0.6.wav", "Gull_5", True)
GULL_7 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/GULL_7.1_0.6.wav", "test_Gull_7", False)

#Great Tits
MUSVIT_1 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/MUSVIT_1_0.6.wav", "Musvit_1", True)
MUSVIT_1_1 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/MUSVIT_1.1_0.6.wav", "Musvit_1.2", True)
MUSVIT_1_2 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/MUSVIT_1.2_0.6.wav", "Musvit_1.3", True)
MUSVIT_3_1 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/MUSVIT_3.1_0.6.wav", "Musvit_3", True)
MUSVIT_3_2 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/MUSVIT_3.2_0.6.wav", "Musvit_3.2", True)
MUSVIT_3_3 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/MUSVIT_3.3_0.6.wav", "test_Musvit_3.3", False)

#Common Chaffinchs
COMMON_1_1 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/COMMON_1.1_0.6.wav", "Common_1.1", True)
COMMON_1_2 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/COMMON_1.2_0.6.wav", "Common_1.2", True)
COMMON_1_3 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/COMMON_1.3_0.6.wav", "Common_1.3", True)
COMMON_2_1 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/COMMON_2.1_0.6.wav", "Common_2.1", True)
COMMON_2_2 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/COMMON_2.2_0.6.wav", "Common_2.2", True)
COMMON_2_3 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/COMMON_2.3_0.6.wav", "Common_2.3_test", False)

#Eurasian Blue Tits
BLUE_TIT_1_1 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/BLUE_TIT_1.1_0.6.wav", "Blue_tit_1.1", True)
BLUE_TIT_1_2 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/BLUE_TIT_1.2_0.6.wav", "Blue_tit_1.2", True)
BLUE_TIT_1_3 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/BLUE_TIT_1.3_0.6.wav", "Blue_tit_1.3", True)
BLUE_TIT_2_1 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/BLUE_TIT_2.1_0.6.wav", "Blue_tit_2.1", True)
BLUE_TIT_2_2 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/BLUE_TIT_2.2_0.6.wav", "Blue_tit_2.2", True)
BLUE_TIT_2_3 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/BLUE_TIT_2.3_0.6.wav", "Blue_tit_2.3_test", False)

#Recorded calls
MUSVIT_optagelse = Fourier("fugle-lyde/Udvalgte_fugle_lyde/MUSVIT_optagelse.wav", "gæt_M.usvit_optagelse", False)

#optaget 12-05-2023
CROW_20_1 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/CROW_20.1_0.6.wav", "gæt_C.row_20_1", False)
CROW_20_2 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/CROW_20.2_0.6.wav", "gæt_C.row_20_2", False)
MUSVIT_20_1 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/MUSVIT_20.1_0.6.wav", "gæt_M.usvit_20_1", False)
MUSVIT_20_2 = Fourier("fugle-lyde/Udvalgte_fugle_lyde/MUSVIT_20.2_0.6.wav", "gæt_M.usvit_20_2", False)

#Create and save spectrograms
#MUSVIT_optagelse.spect() #This is an example of plotting the spectrogram to the recorded Great Tit
"""
CROW_1_1.spect()
CROW_1_2.spect()
CROW_1_3.spect()
CROW_4.spect()
CROW_7.spect()
CROW_8.spect()

GULL_1.spect()
GULL_2.spect()
GULL_3.spect()
GULL_4.spect()
GULL_5.spect()
GULL_7.spect()

MUSVIT_1.spect()
MUSVIT_1_1.spect()
MUSVIT_1_2.spect()
MUSVIT_3_1.spect()
MUSVIT_3_2.spect()
MUSVIT_3_3.spect()

COMMON_1_1.spect()
COMMON_1_2.spect()
COMMON_1_3.spect()
COMMON_2_1.spect()
COMMON_2_2.spect()
COMMON_2_3.spect()

BLUE_TIT_1_1.spect()
BLUE_TIT_1_2.spect()
BLUE_TIT_1_3.spect()
BLUE_TIT_2_1.spect()
BLUE_TIT_2_2.spect()
BLUE_TIT_2_3.spect()

MUSVIT_optagelse.spect()
CROW_20_1.spect()
CROW_20_2.spect()
MUSVIT_20_1.spect()
MUSVIT_20_2.spect()
###################################################################"""

new_PCs =[CROW_8, GULL_7, MUSVIT_3_2, COMMON_2_3, BLUE_TIT_2_3, CROW_20_1, CROW_20_2, MUSVIT_20_1, MUSVIT_20_2, MUSVIT_optagelse] #List of birds NOT in the dataset used for creating the PC-axis

GULL_1.pandaF() #Converts the "global" dataset into a pandas frame to be able to transpose it

data = GULL_1.fugl_pandaframe.T #Transposes the pandas frame to be a row-oriented dataset and saves it

normalized_data = (data - data.mean())

U, S, Vt = np.linalg.svd(normalized_data, full_matrices=False) #Reduced SVD

PCs = np.matmul(U, np.diag(S)) #Matrix-Matrix multiplication

#A list with principal components are created
PC_list = []
for i in S**2:
    PC_list.append(round((i/np.sum(S**2))*100, 2))

labels = [str(i) for i in range(1, len(PC_list)+1)] #Labels PC1, PC2 ... PCn, where n is the number of birds, are created
PCA_df = pd.DataFrame(PCs, index=data.index, columns=labels) #A pandas dataframe where each column is named PC1, PC2, etc. and bird names are used as indexes, is created
#print(PCA_df.iloc[:,[0,1]]) #Prints the first 2 principal components from the dataset
for bird in new_PCs: #Converts the birds not in the dataset into the PC-axis and adds them to the dataset
    PCA_df.loc[bird.fugl] = np.matmul((bird.Zxx_dB_flat.T - data.mean()), Vt.T)

#########################################################
index_List = [] #List for having the order of which index of a bird is furthest away from the last element in the "new_PCs"-list
lighed = [] #List containing the corresponding distance to each birds index in the "index_List"-list

for I in range(len(PC_list)): #Go through each bird in original dataset
    gæt = 10**20 #Random big number needed to start the loop
    for i in range(len(PC_list)): #Goes through all rows
        L=[] #Makes L an empty list (temporary list)
        for k in range(2): #Goes through the first 2 PCs
            L.append((PCA_df.iloc[-1][k]-PCA_df.iloc[i][k])**2)#*(PC_list[k]/100)) #Appends the difference of each PC (loop decides how many PC's) between our guess bird and the actual birds coordinates times the importance of that PC, to L
        LL = np.sqrt(sum(L)) #Finds the length between the 2 points and appoints it to LL
        if i not in index_List and LL < gæt: #Looks if "i" already is in the list of indexes and if LL is lesser than the current shortest distance
            gæt = LL #Saves the shortest distance
            index = i #Saves the index of the shortest distance
    lighed.append(gæt) #Appends the distance
    index_List.append(index) #Appends the best match in index_List, so the same bird is not appended again

for i in range(len(index_List)): #Prints a table showing how good a match each bird is
    print("-" * 50)
    print(f"{i+1:2}. best match: {PCA_df.index.values[index_List[i]]:18} - {lighed[i]/lighed[-1]*100:10.2f}% |") # tallet beskriver procentvise afvigelse fra den største
print("-" * 50)
#########################################################

f, ax = plt.subplots(1, 1, figsize=(16, 5)) #A figure with 1 plot
ax.plot(range(1, len(PC_list)+1), PC_list, color="brown", marker="o", mfc="w") #Using marker and markerfacecolor to plot a circle at every point
ax.set_xticks(range(1, len(PC_list)+1))
ax.set_xticklabels(labels)
ax.xaxis.set_tick_params(labelsize=14) #Font-size of tick_labels on the xaxis
ax.yaxis.set_tick_params(labelsize=14)
ax.set_ylabel("Percentage of Explained Variance", size=14)
ax.set_xlabel("Principal Component", size=14)
ax.set_title("Scree plot", size=16)
#plt.savefig("Scree plot.png", dpi=500)
plt.show() #Shows plot then resets it

#Numbers for bird entries
crow_count = 0
gull_count = 0
great_count = 0
common_count = 0
blue_count = 0
title_count = 0
f, ax = plt.subplots(1, 1, figsize=(14, 5)) #A figure with 1 plot
for fugl in range(0, len(PCA_df.index), 1): #1 point for each bird with fitting color and label
    if "Crow" in PCA_df.index.values[fugl]: #If the bird is known a Crow, it's color and label is changed accordingly
        color = "#222222"
        label = "Hooded Crow"
        crow_count += 1
        ax.text(PCA_df.iloc[fugl][0]-10, PCA_df.iloc[fugl][1]+50, crow_count)
    elif "Gull" in PCA_df.index.values[fugl]: #If it's a Gull
        color = "#00BB00"
        label = "Herring Gull"
        gull_count += 1
        ax.text(PCA_df.iloc[fugl][0]-10, PCA_df.iloc[fugl][1]+50, gull_count)
    elif "Musvit" in PCA_df.index.values[fugl]: #If Great Tit
        color = "#BB0000"
        label = "Great Tit"
        great_count += 1
        ax.text(PCA_df.iloc[fugl][0]-10, PCA_df.iloc[fugl][1]+50, great_count)
    elif "Common" in PCA_df.index.values[fugl]: #If Common Chaffinch
        color = "#BBBB00"
        label = "Common Chaffinch"
        common_count += 1
        ax.text(PCA_df.iloc[fugl][0]-10, PCA_df.iloc[fugl][1]+50, common_count)
    elif "Blue_tit" in PCA_df.index.values[fugl]: #If Eurasian Blue Tit
        color = "#00BBBB"
        label = "Eurasian Blue Tit"
        blue_count += 1
        ax.text(PCA_df.iloc[fugl][0]-10, PCA_df.iloc[fugl][1]+50, blue_count)
    else: #If the bird is unknown
        color = "#454545"
        label = "Unknown"
    if "gæt" in PCA_df.index.values[fugl]: #If the bird is supposed to be guessed it's color and label changes to something else
        color = "#0000BB"
        label = "Recorded"
        title_count += 1
        ax.text(PCA_df.iloc[fugl][0]-10, PCA_df.iloc[fugl][1]+50, title_count)
    elif "test" in PCA_df.index.values[fugl]: #If the bird is known, but is not in the original dataset it's label changes, but not the color
        label = "Test"
        #ax.text(PCA_df.iloc[fugl][0], PCA_df.iloc[fugl][1], label) #But the bird is annotated it's label, to show that it is not in the dataset
    ax.scatter([PCA_df.iloc[fugl][0]], [PCA_df.iloc[fugl][1]], label=label, c=color) #Points are plotted using their label and color
#Styling the plot
ax.set_xlabel(f"PC1 - {PC_list[0]} %")
ax.set_ylabel(f"PC2 - {PC_list[1]} %")
ax.grid()
ax.axhline(y=0, color="k")
ax.axvline(x=0, color="k")

#Create a legend with each colors meaning
handles, labels = plt.gca().get_legend_handles_labels()
while "Test" in labels: #Removes the "Test"-labels and their handles, since they have no unique color and are annotated text
    temp = labels.index("Test")
    labels.pop(temp)
    handles.pop(temp)
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.0, 1.0)) #Create the legend box

ax.set_title("Score plot")
plt.tight_layout()
#plt.savefig("bird_call_compr_scoreplot_3_numbering", dpi=500)
plt.show()
