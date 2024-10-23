
import numpy as np
new_outlier_scores = []
for val in range(101) :
    new_outlier_scores.append(val+1)
    # print(val+1)
print(new_outlier_scores)
median_of_OS = np.median(new_outlier_scores) 
print(median_of_OS)
# for val in new_outlier_scores :
#     print(abs(new_outlier_scores - median_of_OS))
OS_minus_median = abs(new_outlier_scores - median_of_OS)
print(OS_minus_median)
print(OS_minus_median[50])
median_of_OS_minus_median = np.median(abs(new_outlier_scores - median_of_OS))
print(median_of_OS_minus_median)
