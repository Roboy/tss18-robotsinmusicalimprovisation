import pypianroll as ppr
import glob

"""CHECK VALIDITY OF MIDIFILES"""

testValidity = glob.glob("../DougMcKenzieFiles/test/*.mid")

for i, file in enumerate(testValidity):
    print(i);print(file);
    ppr.Multitrack(file, beat_resolution=24).get_stacked_pianorolls()
    
    
print('done')