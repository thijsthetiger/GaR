
'''
Dynamic GaR (Growth at Risk) module for Python
==============================================

GAR is a Python module for running Growth at Risk calculations.
The user will use the provided Excel template to fill the data
and choose options, then run the code by clicking on Excel buttons.

For assistance contact
XXX YYY (XXX.imf.org)
'''

__version__ = "0.1.1"

import os
import sys
import importlib
import importlib.util

try:
    import xlwings as xw
except ModuleNotFoundError as error:
    if error.name != 'xlwings':
        raise
    xw = None

# Initialize the wb to None
wb = None

_excel_automation_available = (
    xw is not None
    and importlib.util.find_spec('win32api') is not None
    and importlib.util.find_spec('win32con') is not None
)

if xw is not None:
    # Set the wb from the caller if this library has been imported through Excel.
    try:
        wb = xw.Book.caller()
    except Exception:
        # Importing from a regular Python environment has no caller workbook.
        print('-' * 50)
        print('GAR WARNING:')
        print('    When importing the GAR library from a non-Excel')
        print('    environment it is necessary to call')
        print("    GAR.set_mock_caller_file(xlfilename)")
        print('-' * 50)

if _excel_automation_available:
    from GAR.partition.partition_main import do_partition
    from GAR.quantfit.quantfit_main import do_quantfit
    from GAR.tsfit.tsfit_main import do_tsfit
    from GAR.scenario.scenariomain import do_scenario
    from GAR.historical.historicalmain import do_historical
    from GAR.segment.segmentmain import do_segment
    from GAR import historical
    from GAR import scenario
    from GAR import segment
else:
    def _require_excel_automation(*args, **kwargs):
        raise ImportError(
            'The original IMF Excel workflow requires xlwings and Windows '
            'pywin32. Use run_narrative_gar.py for the active thesis models.'
        )

    do_partition = _require_excel_automation
    do_quantfit = _require_excel_automation
    do_tsfit = _require_excel_automation
    do_scenario = _require_excel_automation
    do_historical = _require_excel_automation
    do_segment = _require_excel_automation


def set_mock_caller_file(xlfilename='GaR.xlsm'):
    '''
    Function to set up mock caller for xlwings.
    This function needs to be called with the input Excel filename
    when trying to run the GaR package from the terminal.

    **When running from the input Excel this should not be called.**
    '''

    if not _excel_automation_available:
        raise ImportError(
            'The original IMF Excel workflow requires xlwings and Windows '
            'pywin32.'
        )

    # Set the wb set in this function to be global
    # so it overwrites the wb throughout this file.
    global wb

    # Get the full path of the specified file
    xlfullpath = os.path.abspath(xlfilename)
    if not os.path.isfile(xlfullpath):
        print('File ' + xlfullpath + ' does not exist')
        sys.exit(-1)

    xw.Book(xlfullpath).set_mock_caller()
    wb = xw.Book.caller()
    print('Set file ' + xlfullpath + ' to be mock caller')

    # Once this is called we need to re-import the functions,
    # otherwise the variable wb imported is still the previous value
    importlib.reload(globals)
    
    importlib.reload(partition.partition_main)
    from GAR.partition.partition_main import do_partition
    
    importlib.reload(quantfit.quantfit_main)
    from GAR.quantfit.quantfit_main import do_quantfit

    importlib.reload(tsfit.tsfit_main)
    from GAR.tsfit.tsfit_main import do_tsfit  
        
    importlib.reload(scenario.scenariomain)
    from GAR.scenario.scenariomain import do_scenario

    importlib.reload(historical.historicalmain)
    from GAR.historical.historicalmain import do_historical
    
    importlib.reload(segment.segmentmain)
    from GAR.segment.segmentmain import do_segment
