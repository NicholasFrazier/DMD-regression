# DMD-regression
Johnnie Jones package for DMD homework on regression

There are modifications to what the TA provided so the code uses "Python classes differently".  I use "smf" in the import so that I can leverage something called statsmodels.formula.api.  You will see that I commented out the call to the OLS class and created my own call with a "Q" function that is used when the variable name has spaces in it.  Enjoy!!!

import statsmodels.api as sm
import statsmodels.formula.api as smf  # Import statsmodels.formula.api
