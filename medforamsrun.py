from parcels import FieldSet, Field, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, Variable
from datetime import timedelta as delta
from datetime import datetime as datetime
from glob import glob
import numpy as np
import xarray as xr

wstokes = False
ddir = '/Users/erik/Downloads/'
nemofiles = {'U': ddir+'sv03-med-ingv-cur-rean-d_1564143903715.nc',
             'V': ddir+'sv03-med-ingv-cur-rean-d_1564143903715.nc',
             'T': ddir+'sv03-med-ingv-tem-rean-d_1564143623651.nc',
             'S': ddir+'sv03-med-ingv-sal-rean-d_1564144045564.nc'}

nemovariables = {'U': 'vozocrtx', 'V': 'vomecrty', 'T': 'votemper', 'S': 'vosaline'}
nemodimensions = {'lon': 'lon', 'lat': 'lat', 'time': 'time'}
fieldset = FieldSet.from_nemo(nemofiles, nemovariables, nemodimensions)

sec_startlat = [34.084706, 34.115607, 34.910005, 35.050183, 34.971962, 34.979806, 37.501224, 38.252642, 35.305275, 34.766999, 34.851108, 35.755278, 36.657987, 36.998696, 37.226809, 37.665342, 37.988935, 38.501585, 39.324758, 38.750331, 38.492781, 37.442948, 36.978753, 36.316667, 36.266667]
sec_startlon = [27.840839, 27.582864, 24.695936, 20.633217, 18.184907, 18.03555, 17.006161, 16.718924, 14.466557, 14.183237, 14.163567, 13.893837, 13.603665, 13.47317, 13.054473, 12.181478, 11.509662, 8.992273, 5.483495, 3.750185, 2.693403, -1.175272, -1.340577, -3.980977, -4.289487]
sec_endlat = [34.115607, 34.090673, 34.132291, 35.039292, 34.964814, 34.960507, 38.252642, 38.342401, 34.766999, 34.851108, 35.755278, 36.657987, 36.998696, 37.226809, 37.665342, 37.991648, 38.087668, 38.500205, 39.4419, 38.781292, 38.24337, 37.410175, 36.749641, 36.266667, 36.05]
sec_endlon = [27.582864, 27.065639, 23.449737, 20.207551, 18.038343, 18.033557, 16.718924, 17.358082, 14.183237, 14.163567, 13.893837, 13.603665, 13.47317, 13.054473, 12.181478, 11.51077, 11.250037, 8.796319, 5.012563, 3.833372, 1.723399, -1.07361, -1.499191, -4.289487, -6.2311]
# sec_names = ['406PP10', '406PP11', '406PP16', '406PP21', '406PP25', '406PP28', '406PP32', '406PP33', '407PP03', '407PP04', '407PP05', '407PP06', '407PP07', '407PP08', '407PP09', '407PP10', '407PP12', '407PP18', '407PP24', '407PP27', '407PP31', '407PP41', '407PP43', '407PP49', '407PP50']
sec_dates = ['17-jan-16', '17-jan-16', '19-jan-16', '20-jan-16', '21-jan-16', '22-jan-16', '23-jan-16', '23-jan-16', '11-feb-16', '11-feb-16', '12-feb-16', '12-feb-16', '12-feb-16', '12-feb-16', '13-feb-16', '13-feb-16', '13-feb-16', '15-feb-16', '16-feb-16', '17-feb-16', '18-feb-16', '20-feb-16', '21-feb-16', '23-feb-16', '23-feb-16']

npersec = 100

fieldset.T.interp_method='nearest'
fieldset.S.interp_method='nearest'

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def SampleTS(particle, fieldset, time):
    if (fieldset.T[time, particle.depth, particle.lat-0.0834, particle.lon] > 0) and \
    (fieldset.T[time, particle.depth, particle.lat+0.0834, particle.lon] > 0) and \
    (fieldset.T[time, particle.depth, particle.lat, particle.lon+0.0834] > 0) and \
    (fieldset.T[time, particle.depth, particle.lat, particle.lon-0.0834] > 0):
        particle.temp = fieldset.T[time, particle.depth, particle.lat, particle.lon]
        particle.salt = fieldset.S[time, particle.depth, particle.lat, particle.lon]
    else:
        particle.temp = -9999
        particle.salt = -9999

def Age(fieldset, particle, time):
    particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > 30*86400:
        particle.delete()


class ForamParticle(JITParticle):
    temp = Variable('temp', initial=fieldset.T)
    salt = Variable('salt', initial=fieldset.S)
    age = Variable('age', initial = 0.)

pset = ParticleSet.from_line(fieldset=fieldset, pclass=ForamParticle,
                             start=(sec_startlon[0], sec_startlat[0]),
                             finish=(sec_endlon[0], sec_endlat[0]), size=npersec,
                             time=datetime.strptime(sec_dates[0], '%d-%b-%y'))

for s in range(1, len(sec_startlat)-1):
    pset.add(ParticleSet.from_line(fieldset=fieldset, pclass=ForamParticle,
                             start=(sec_startlon[s], sec_startlat[s]),
                             finish=(sec_endlon[s], sec_endlat[s]), size=npersec,
                             time=datetime.strptime(sec_dates[s], '%d-%b-%y')))

outfile = pset.ParticleFile(name="medforams.nc", outputdt=delta(days=1))

pset.execute(AdvectionRK4 +pset.Kernel(SampleTS) + Age, dt=delta(hours=-1), 
             output_file=outfile, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
outfile.close()
