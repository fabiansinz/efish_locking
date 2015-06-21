from pprint import pprint
import re
import datajoint as dj
from datajoint import schema
import glob
import yaml
BASEDIR = '/home/fabee/data/carolin/'
server = schema('efish',locals())

def scan_info(cell_id):
    """
    Scans the info.dat for meta information about the recording.

    :param cell_id: id of the cell
    :return: meta information about the recording as a dictionary.
    """
    info = open(BASEDIR + cell_id + '/info.dat').readlines()
    info = [re.sub(r'[^\x00-\x7F]+',' ', e[1:]) for e in info]
    meta = yaml.load(''.join(info))
    return meta


@server
class EFishes(dj.Manual):
    definition = """
    # Basics weakly electric fish subject info

    fish_id                                 : varchar(40)     # unique fish id
    ---

    eod_frequency                           : float # EOD frequency in Hz
    species = "Apteronotus leptorhynchus"   : enum('Apteronotus leptorhynchus', 'Eigenmannia virescens') # species
    gender  = "unknown"                      : enum('unknown', 'make', 'female') # gender
    weight : float # weight in g
    size: float  # size in cm
    """

    def make_tuples(self):
        for (cell,) in PaperCells().fetch():
            a = scan_info(cell)
            a = a['Subject'] if 'Subject' in a else a['Recording']['Subject']
            dat = {'fish_id': a['Identifier'],
                   'eod_frequency': float(a['EOD Frequency'][:-2]),
                   'gender': a['Gender'].lower(),
                   'weight': float(a['Weight'][:-1]),
                   'size': float(a['Size'][:-2])}
            try:
                self.insert(dat)
            except Exception as e:
                print(e)

@server
class PaperCells(dj.Lookup):
    definition = """
    # What cell ids make it into the paper

    cell_id                          : varchar(40)      # unique cell id
    ---
    """

    def __init__(self):
        dj.Lookup.__init__(self)

    def make_tuples(self):
        cells = [{'cell_id': '2014-12-03-al'},
                 {'cell_id': '2014-07-23-ae'},
                 {'cell_id': '2014-12-11-ae-invivo-1'},
                 {'cell_id': '2014-12-11-aj-invivo-1'},
                 {'cell_id': '2014-12-11-al-invivo-1'},
                 {'cell_id': '2014-09-02-ad'},
                 {'cell_id': '2014-12-03-ab'},
                 {'cell_id': '2014-07-23-ai'},
                 {'cell_id': '2014-12-03-af'},
                 {'cell_id': '2014-07-23-ab'},
                 {'cell_id': '2014-07-23-ah'},
                 {'cell_id': '2014-11-13-aa'},
                 {'cell_id': '2014-11-26-ab'},
                 {'cell_id': '2014-11-26-ad'},
                 {'cell_id': '2014-10-29-aa'},
                 {'cell_id': '2014-12-03-ae'},
                 {'cell_id': '2014-09-02-af'},
                 {'cell_id': '2014-12-11-ah-invivo-1'},
                 {'cell_id': '2014-07-23-ad'},
                 {'cell_id': '2014-12-11-ac-invivo-1'},
                 {'cell_id': '2014-12-11-ab-invivo-1'},
                 {'cell_id': '2014-12-11-ag-invivo-1'},
                 {'cell_id': '2014-12-03-ad'},
                 {'cell_id': '2014-12-11-ak-invivo-1'},
                 {'cell_id': '2014-09-02-ag'},
                 {'cell_id': '2014-12-03-ao'},
                 {'cell_id': '2014-12-03-aj'},
                 {'cell_id': '2014-07-23-aj'},
                 {'cell_id': '2014-11-26-ac'},
                 {'cell_id': '2014-12-03-ai'},
                 {'cell_id': '2014-06-06-ak'},
                 {'cell_id': '2014-11-13-ab'},
                 {'cell_id': '2014-05-21-ab'},
                 {'cell_id': '2014-07-23-ag'},
                 {'cell_id': '2014-12-03-ah'},
                 {'cell_id': '2014-07-23-aa'},
                 {'cell_id': '2014-12-11-am-invivo-1'},
                 {'cell_id': '2014-12-11-aa-invivo-1'}]
        for c in cells:
            try:
                self.insert(c)
            except Exception as e:
                print(e)

@server
class Cells(dj.Imported):
    definition = """
    # Recorded cell with additional info

    ->PaperCells                                # cell id to be imported
    ---
    ->EFishes                                   # fish this cell was recorded from
    recording_date                   : date     #  recording date
    cell_type                        : enum('p-unit', 'i-cell', 'e-cell','ampullary','ovoid','unkown') # cell type
    recording_location               : enum('nerve', 'ell') # where
    depth                            : float    # recording depth in mu
    baseline                         : float    # baseline firing rate in Hz
    """

    @property
    def populate_relation(self):
        return PaperCells()

    def _make_tuples(self, key):

        a = scan_info(key['cell_id'])
        subj = a['Subject'] if 'Subject' in a else a['Recording']['Subject']
        cl = a['Cell'] if 'Cell' in a else a['Recording']['Cell']
        dat = {'cell_id': key['cell_id'],
               'fish_id': subj['Identifier'],
               'recording_date': a['Recording']['Date'],
               'cell_type': cl['CellType'].lower(),
               'recording_location': 'nerve'  if cl['Structure'].lower() == 'nerve' else 'ell',
               'depth': float(cl['Depth'][:-2]),
               'baseline': float(cl['Cell properties']['Firing Rate1'][:-2])
               }
        self.insert(dat)

if __name__=="__main__":
    pc = PaperCells()
    pc.make_tuples()

    ef = EFishes()
    ef.make_tuples()

    cl = Cells()
    cl.populate()
    print(cl)