import pandas as pd

def create_new_location(row):
    # Check if the location contains "jakarta" or "dki jakarta" or "jakarta pusat"
    location = row['location'].lower()

    #jawa
    if 'jogjakarta' in location or 'yogyakarta' in location or 'jogja' in location or 'jogyakarta' in location:
        return 'yogyakarta'
    elif 'jakarta' in location or 'dki jakarta' in location or 'jakarta pusat' in location or 'kebayoran' in location or 'menteng' in location:
        return 'jakarta'
    elif 'jawa barat' in location or 'jabar' in location  or 'garut' in location  or 'bekasi' in location or 'bogor' in location or 'bandung' in location or 'subang' in location or 'ciledug' in location or 'cirebon' in location:
        return 'jawa barat'
    elif 'tangerang' in location or 'banten' in location or 'serang' in location :
        return 'banten'
    elif 'jawa tengah' in location or 'semarang' in location or 'jateng' in location  or 'purwodadi' in location  or 'banjarnegara' in location or 'purworejo'in location or 'slawi' in location or 'pekalongan' in location or 'kebumen' in location or 'cilacap' in location or 'solo' in location or 'banyumas' in location or 'purwokerto' in location or 'jepara' in location:
        return 'jawa tengah'
    elif 'jawa timur' in location  or 'jatim' in location or 'malang' in location or 'surabaya' in location or 'madiun' in location or 'ponorogo' in location:
        return 'jawa timur'

    #bali
    elif 'bali' in location or 'denpasar' in location or 'kuta' in location:
        return 'bali'

    #Sumatra
    elif 'aceh' in location or 'nanggroe aceh darussalam' in location:
        return 'aceh'
    elif 'sumatera utara' in location or 'medan' in location :
        return 'sumatera utara'
    elif 'sumatera selatan' in location or 'palembang' in location :
        return 'sumatera selatan'
    elif 'bengkulu' in location:
        return 'bengkulu'
    elif 'kepulauan riau' in location or 'tanjung pinang' in location or 'batam' in location:
        return 'kepulauan riau'
    elif 'sumatera barat' in location or 'padang' in location :
        return 'sumatera barat'
    elif 'jambi' in location or 'malang' in location:
        return 'jambi'
    elif 'lampung' in location or 'bandar lampung' in location :
        return 'lampung'
    elif 'bangka belitung' in location or 'pangkal pinang' in location :
        return 'bangka belitung'
    elif 'riau' in location or 'pekan baru' in location:
        return 'riau'

    #kalimantan
    elif 'kalimantan barat' in location or 'pontianak' in location:
        return 'kalimantan barat'
    elif 'Kalimantan Tengah' in location or 'palangkaraya' in location:
        return 'kalimantan tengah'
    elif 'kalimantan selatan' in location or 'banjarmasin' in location:
        return 'kalimantan selatan'
    elif 'kalimantan timur' in location or 'samarinda' in location:
        return 'kalimantan timur'
    elif 'kalimantan utara' in location or 'tanjung selor' in location:
        return 'kalimantan utara'


    #sulawesi
    elif 'sulawesi utara' in location or 'manado' in location:
        return 'sulawesi utara'
    elif 'sulawesi tengah' in location or ' palu' in location:
        return 'sulawesi tengah'
    elif 'sulawesi selatan' in location or 'makassar' in location:
        return 'sulawesi selatan'
    elif 'sulawesi tenggara' in location or 'kendari' in location:
        return 'sulawesi tenggara'
    elif 'sulawesi barat' in location or 'mamuju' in location:
        return 'sulawesi barat'
    elif 'gorontalo' in location :
        return 'gorontalo'

    #Nusa Tenggara
    elif 'nusa tenggara barat' in location or 'mataram' in location or 'lombok' in location or 'sumbawa' in location:
        return 'nusa tenggara barat'
    elif 'nusa tenggara timur' in location or 'kupang' in location:
        return 'nusa tenggara timur'

    #papua
    elif 'papua selatan' in location:
        return 'papua selatan'
    elif 'papua tengah' in location:
        return 'papua tengah'
    elif 'papua pegunungan' in location:
        return 'papua pegunungan'
    elif 'papua barat daya' in location:
        return 'papua barat daya'

    # elif 'indonesia' in location or 'nusantara' in location:
    #     return 'indonesia'
    else:
	
        return 'unknown'




