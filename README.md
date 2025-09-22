This file is the R file for a visitation prediction algorithm for zoos victoria. After the arrival of elephants it became outdated however, it had a reasonably high confidence percentage which was useful in predicting staffing requirements. 

Please note that this is a final product of many iterations and feature selection which landed on the current set of features.

Some extra pre processing of the data would be useful and perhaps some additional parameters. 

I would pre process the data better by removing larger group books(schools and holiday programs) from visitation data as their attendence is usually pre determinded and not weather dependent.

I would add the parameter of OKTA cloud cover instead of solar MJ/m^2 and even use forecasted weather instead of assuming the actual weather is an accurate repensentation of forecasted weather. 

I would also consider time since/until next school holidays as a new parameter, this was previously untested.
