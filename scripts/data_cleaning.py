import tqdm
import re
import numpy as np # linear algebra
import pandas as pd # data processing,
from pandarallel import pandarallel
import nltk
import emoji
from nltk import sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer

nltk.download('punkt')
tokenizer = TreebankWordTokenizer()

LANGS = {
    'en': 'english',
    'it': 'italian', 
    'fr': 'french', 
    'es': 'spanish',
    'tr': 'turkish', 
    'ru': 'russian',
    'pt': 'portuguese'
}

def clean_text(text, lang='en'):
    try:
        text = str(text)
        text = re.sub(r'[0-9"]', '', text)
        text = re.sub(r'#[\S]+\b', '', text)
        text = re.sub(r'@[\S]+\b', '', text)
        text = re.sub(r'https?\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub(r'\s+', ' ', text)
        # text = re.sub("\[\[User.*",'', text)
        # text = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'', text)
        text = exclude_duplicate_sentences(text, lang)
    except:
        print(f'Exception occured: {text}, {type(text)}')
        raise

    return text.strip()

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', '█', '½', '…', '\xa0', '\t',
 '“', '★', '”', '–', '●', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲', '¸', '¾', '⋅', '‘', '∞', '«',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]


mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"couldnt" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"doesnt" : "does not",
"don't" : "do not",
"don\x89Ûªt": "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"havent" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"I'm": "I am",
"I\x89Ûªm": "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"shouldnt" : "should not",
"that's" : "that is",
"thats" : "that is",
"there's" : "there is",
"theres" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"theyre":  "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"you\x89Ûªve": "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}

def clean_puncts(x):
    x = str(x).replace("\n","")
    for punct in puncts:
        x = x.replace(punct, f"")
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def handle_contractions(x, lang):
    if lang == 'en':
        x = tokenizer.tokenize(x)
    return x

def fix_quote(x):
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


def replace_typical_misspell(text, lang):
    if lang == 'en':
        mispellings, mispellings_re = _get_mispell(mispell_dict)
    
        def replace(match):
            return mispellings[match.group(0)]
    
        return mispellings_re.sub(replace, text)
    else:
        return text

def get_sentences(text, lang='en'):
    return sent_tokenize(str(text), LANGS.get(lang, 'english'))

def exclude_duplicate_sentences(text, lang='en'):
    sentences = []
    for sentence in get_sentences(text, lang):
        sentence = sentence.strip()
        if sentence not in sentences:
            sentences.append(sentence)
    
    return ' '.join(sentences)


def clean_data(df, columns: list):
    for col in columns:
        if col in df.columns:
            # df[col] = df[col].apply(lambda x: clean_numbers(x))
            df[col] = df[[col, 'lang']].parallel_apply(lambda x: clean_text(x[col], x['lang']), axis=1)
            # df[col] = df[[col, 'lang']].parallel_apply(lambda x: replace_typical_misspell(x[col], x['lang']), axis=1)
            # df[col] = df[[col, 'lang']].parallel_apply(lambda x: handle_contractions(x[col], x['lang']), axis=1)  
            # # df[col] = df[col].parallel_apply(lambda x: fix_quote(x))
            # df[col] = df[[col, 'lang']].parallel_apply(lambda x: clean_non_latin(x[col], x['lang']), axis=1)
            # df[col] = df[col].parallel_apply(lambda x: clean_puncts(x.lower())) 
    
    return df


# ****** Twitter specific functs ******
# ***** Twitter *****
# Special characters
twitter_special_chars = {
"\x89Û_":"",
"\x89ÛÒ": "",
"\x89ÛÓ": "",
"\x89ÛÏWhen": "When",
"\x89ÛÏ": "",
"China\x89Ûªs": "China's",
"let\x89Ûªs": "let's",
"\x89Û÷": "",
"\x89Ûª": "",
"\x89Û\x9d": "",
"å_": "",
"\x89Û¢": "",
"\x89Û¢åÊ": "",
"fromåÊwounds": "from wounds",
"åÊ": "",
"åÈ": "",
"JapÌ_n": "Japan",
"Ì©": "e",
"å¨": "",
"SuruÌ¤": "Suruc",
"åÇ": "",
"å£3million": "3 million",
"åÀ": "",
}

# Character entity references
twitter_character_entity = {
"&gt;": ">",
"&lt;": "<",
"&amp;": "&",
}

# Typos, slang and informal abbreviations
twitter_typos_slang ={
"w/e": "whatever",
"w/": "with",
"USAgov": "USA government",
"recentlu": "recently",
"Ph0tos": "Photos",
"amirite": "am I right",
"exp0sed": "exposed",
"<3": "love",
"amageddon": "armageddon",
"Trfc": "Traffic",
"8/5/2015": "2015-08-05",
"WindStorm": "Wind Storm",
"8/6/2015": "2015-08-06",
"10:38PM": "10:38 PM",
"10:30pm": "10:30 PM",
"16yr": "16 year",
"lmao": "laughing my ass off",   
"TRAUMATISED": "traumatized", 
}

# Hashtags and usernames
twitter_hashtag={"IranDeal":"Iran Deal",
"ArianaGrande":"Ariana Grande",
"camilacabello97":"camila cabello",
"RondaRousey":"Ronda Rousey",
"MTVHottest":"MTV Hottest",
"TrapMusic":"Trap Music",
"ProphetMuhammad":"Prophet Muhammad",
"PantherAttack":"Panther Attack",
"StrategicPatience":"Strategic Patience",
"socialnews":"social news",
"NASAHurricane":"NASA Hurricane",
"onlinecommunities":"online communities",
"humanconsumption":"human consumption",
"Typhoon-Devastated":"Typhoon Devastated",
"Meat-Loving":"Meat Loving",
"facialabuse":"facial abuse",
"LakeCounty":"Lake County",
"BeingAuthor":"Being Author",
"withheavenly":"with heavenly",
"thankU":"thank you",
"iTunesMusic":"iTunes Music",
"OffensiveContent":"Offensive Content",
"WorstSummerJob":"Worst Summer Job",
"HarryBeCareful":"Harry Be Careful",
"NASASolarSystem":"NASA Solar System",
"animalrescue":"animal rescue",
"KurtSchlichter":"Kurt Schlichter",
"aRmageddon":"armageddon",
"Throwingknifes":"Throwing knives",
"GodsLove":"God's Love",
"bookboost":"book boost",
"ibooklove":"I book love",
"NestleIndia":"Nestle India",
"realDonaldTrump":"Donald Trump",
"DavidVonderhaar":"David Vonderhaar",
"CecilTheLion":"Cecil The Lion",
"weathernetwork":"weather network",
"withBioterrorism&use":"with Bioterrorism & use",
"Hostage&2":"Hostage & 2",
"GOPDebate":"GOP Debate",
"RickPerry":"Rick Perry",
"frontpage":"front page",
"NewsInTweets":"News In Tweets",
"ViralSpell":"Viral Spell",
"til_now":"until now",
"volcanoinRussia":"volcano in Russia",
"ZippedNews":"Zipped News",
"MicheleBachman":"Michele Bachman",
"53inch":"53 inch",
"KerrickTrial":"Kerrick Trial",
"abstorm":"Alberta Storm",
"Beyhive":"Beyonce hive",
"IDFire":"Idaho Fire",
"DETECTADO":"Detected",
"RockyFire":"Rocky Fire",
"Listen/Buy":"Listen / Buy",
"NickCannon":"Nick Cannon",
"FaroeIslands":"Faroe Islands",
"yycstorm":"Calgary Storm",
"IDPs:":"Internally Displaced People :",
"ArtistsUnited":"Artists United",
"ClaytonBryant":"Clayton Bryant",
"jimmyfallon":"jimmy fallon",
"justinbieber":"justin bieber",
"UTC2015":"UTC 2015",
"Time2015":"Time 2015",
"djicemoon":"dj icemoon",
"LivingSafely":"Living Safely",
"FIFA16":"Fifa 2016",
"thisiswhywecanthavenicethings":"this is why we cannot have nice things",
"bbcnews":"bbc news",
"UndergroundRailraod":"Underground Railraod",
"c4news":"c4 news",
"OBLITERATION":"obliteration",
"MUDSLIDE":"mudslide",
"NoSurrender":"No Surrender",
"NotExplained":"Not Explained",
"greatbritishbakeoff":"great british bake off",
"LondonFire":"London Fire",
"KOTAWeather":"KOTA Weather",
"LuchaUnderground":"Lucha Underground",
"KOIN6News":"KOIN 6 News",
"LiveOnK2":"Live On K2",
"9NewsGoldCoast":"9 News Gold Coast",
"nikeplus":"nike plus",
"david_cameron":"David Cameron",
"peterjukes":"Peter Jukes",
"JamesMelville":"James Melville",
"megynkelly":"Megyn Kelly",
"cnewslive":"C News Live",
"JamaicaObserver":"Jamaica Observer",
"TweetLikeItsSeptember11th2001":"Tweet like it is september 11th 2001",
"cbplawyers":"cbp lawyers",
"fewmoretweets":"few more tweets",
"BlackLivesMatter":"Black Lives Matter",
"cjoyner":"Chris Joyner",
"ENGvAUS":"England vs Australia",
"ScottWalker":"Scott Walker",
"MikeParrActor":"Michael Parr",
"4PlayThursdays":"Foreplay Thursdays",
"TGF2015":"Tontitown Grape Festival",
"realmandyrain":"Mandy Rain",
"GraysonDolan":"Grayson Dolan",
"ApolloBrown":"Apollo Brown",
"saddlebrooke":"Saddlebrooke",
"TontitownGrape":"Tontitown Grape",
"AbbsWinston":"Abbs Winston",
"ShaunKing":"Shaun King",
"MeekMill":"Meek Mill",
"TornadoGiveaway":"Tornado Giveaway",
"GRupdates":"GR updates",
"SouthDowns":"South Downs",
"braininjury":"brain injury",
"auspol":"Australian politics",
"PlannedParenthood":"Planned Parenthood",
"calgaryweather":"Calgary Weather",
"weallheartonedirection":"we all heart one direction",
"edsheeran":"Ed Sheeran",
"TrueHeroes":"True Heroes",
"S3XLEAK":"sex leak",
"ComplexMag":"Complex Magazine",
"TheAdvocateMag":"The Advocate Magazine",
"CityofCalgary":"City of Calgary",
"EbolaOutbreak":"Ebola Outbreak",
"SummerFate":"Summer Fate",
"RAmag":"Royal Academy Magazine",
"offers2go":"offers to go",
"foodscare":"food scare",
"MNPDNashville":"Metropolitan Nashville Police Department",
"TfLBusAlerts":"TfL Bus Alerts",
"GamerGate":"Gamer Gate",
"IHHen":"Humanitarian Relief",
"spinningbot":"spinning bot",
"ModiMinistry":"Modi Ministry",
"TAXIWAYS":"taxi ways",
"Calum5SOS":"Calum Hood",
"po_st":"po.st",
"scoopit":"scoop.it",
"UltimaLucha":"Ultima Lucha",
"JonathanFerrell":"Jonathan Ferrell",
"aria_ahrary":"Aria Ahrary",
"rapidcity":"Rapid City",
"OutBid":"outbid",
"lavenderpoetrycafe":"lavender poetry cafe",
"EudryLantiqua":"Eudry Lantiqua",
"15PM":"15 PM",
"OriginalFunko":"Funko",
"rightwaystan":"Richard Tan",
"CindyNoonan":"Cindy Noonan",
"RT_America":"RT America",
"narendramodi":"Narendra Modi",
"BakeOffFriends":"Bake Off Friends",
"TeamHendrick":"Hendrick Motorsports",
"alexbelloli":"Alex Belloli",
"itsjustinstuart":"Justin Stuart",
"gunsense":"gun sense",
"DebateQuestionsWeWantToHear":"debate questions we want to hear",
"RoyalCarribean":"Royal Carribean",
"samanthaturne19":"Samantha Turner",
"JonVoyage":"Jon Stewart",
"renew911health":"renew 911 health",
"SuryaRay":"Surya Ray",
"pattonoswalt":"Patton Oswalt",
"minhazmerchant":"Minhaz Merchant",
"TLVFaces":"Israel Diaspora Coalition",
"pmarca":"Marc Andreessen",
"pdx911":"Portland Police",
"jamaicaplain":"Jamaica Plain",
"Japton":"Arkansas",
"RouteComplex":"Route Complex",
"INSubcontinent":"Indian Subcontinent",
"NJTurnpike":"New Jersey Turnpike",
"Politifiact":"PolitiFact",
"Hiroshima70":"Hiroshima",
"GMMBC":"Greater Mt Moriah Baptist Church",
"versethe":"verse the",
"TubeStrike":"Tube Strike",
"MissionHills":"Mission Hills",
"ProtectDenaliWolves":"Protect Denali Wolves",
"NANKANA":"Nankana",
"SAHIB":"Sahib",
"PAKPATTAN":"Pakpattan",
"Newz_Sacramento":"News Sacramento",
"gofundme":"go fund me",
"pmharper":"Stephen Harper",
"IvanBerroa":"Ivan Berroa",
"LosDelSonido":"Los Del Sonido",
"bancodeseries":"banco de series",
"timkaine":"Tim Kaine",
"IdentityTheft":"Identity Theft",
"AllLivesMatter":"All Lives Matter",
"mishacollins":"Misha Collins",
"BillNeelyNBC":"Bill Neely",
"BeClearOnCancer":"be clear on cancer",
"Kowing":"Knowing",
"ScreamQueens":"Scream Queens",
"AskCharley":"Ask Charley",
"BlizzHeroes":"Heroes of the Storm",
"BradleyBrad47":"Bradley Brad",
"HannaPH":"Typhoon Hanna",
"meinlcymbals":"MEINL Cymbals",
"Ptbo":"Peterborough",
"cnnbrk":"CNN Breaking News",
"IndianNews":"Indian News",
"savebees":"save bees",
"GreenHarvard":"Green Harvard",
"StandwithPP":"Stand with planned parenthood",
"hermancranston":"Herman Cranston",
"WMUR9":"WMUR-TV",
"RockBottomRadFM":"Rock Bottom Radio",
"ameenshaikh3":"Ameen Shaikh",
"ProSyn":"Project Syndicate",
"Daesh":"ISIS",
"s2g":"swear to god",
"listenlive":"listen live",
"CDCgov":"Centers for Disease Control and Prevention",
"FoxNew":"Fox News",
"CBSBigBrother":"Big Brother",
"JulieDiCaro":"Julie DiCaro",
"theadvocatemag":"The Advocate Magazine",
"RohnertParkDPS":"Rohnert Park Police Department",
"THISIZBWRIGHT":"Bonnie Wright",
"Popularmmos":"Popular MMOs",
"WildHorses":"Wild Horses",
"FantasticFour":"Fantastic Four",
"HORNDALE":"Horndale",
"PINER":"Piner",
"BathAndNorthEastSomerset":"Bath and North East Somerset",
"thatswhatfriendsarefor":"that is what friends are for",
"residualincome":"residual income",
"YahooNewsDigest":"Yahoo News Digest",
"MalaysiaAirlines":"Malaysia Airlines",
"AmazonDeals":"Amazon Deals",
"MissCharleyWebb":"Charley Webb",
"shoalstraffic":"shoals traffic",
"GeorgeFoster72":"George Foster",
"pop2015":"pop 2015",
"_PokemonCards_":"Pokemon Cards",
"DianneG":"Dianne Gallagher",
"KashmirConflict":"Kashmir Conflict",
"BritishBakeOff":"British Bake Off",
"FreeKashmir":"Free Kashmir",
"mattmosley":"Matt Mosley",
"BishopFred":"Bishop Fred",
"EndConflict":"End Conflict",
"EndOccupation":"End Occupation",
"UNHEALED":"unhealed",
"CharlesDagnall":"Charles Dagnall",
"Latestnews":"Latest news",
"KindleCountdown":"Kindle Countdown",
"NoMoreHandouts":"No More Handouts",
"datingtips":"dating tips",
"charlesadler":"Charles Adler",
"twia":"Texas Windstorm Insurance Association",
"txlege":"Texas Legislature",
"WindstormInsurer":"Windstorm Insurer",
"Newss":"News",
"hempoil":"hemp oil",
"CommoditiesAre":"Commodities are",
"tubestrike":"tube strike",
"JoeNBC":"Joe Scarborough",
"LiteraryCakes":"Literary Cakes",
"TI5":"The International 5",
"thehill":"the hill",
"3others":"3 others",
"stighefootball":"Sam Tighe",
"whatstheimportantvideo":"what is the important video",
"ClaudioMeloni":"Claudio Meloni",
"DukeSkywalker":"Duke Skywalker",
"carsonmwr":"Fort Carson",
"offdishduty":"off dish duty",
"andword":"and word",
"rhodeisland":"Rhode Island",
"easternoregon":"Eastern Oregon",
"WAwildfire":"Washington Wildfire",
"fingerrockfire":"Finger Rock Fire",
"57am":"57 am",
"fingerrockfire":"Finger Rock Fire",
"JacobHoggard":"Jacob Hoggard",
"newnewnew":"new new new",
"under50":"under 50",
"getitbeforeitsgone":"get it before it is gone",
"freshoutofthebox":"fresh out of the box",
"amwriting":"am writing",
"Bokoharm":"Boko Haram",
"Nowlike":"Now like",
"seasonfrom":"season from",
"epicente":"epicenter",
"epicenterr":"epicenter",
"sicklife":"sick life",
"yycweather":"Calgary Weather",
"calgarysun":"Calgary Sun",
"approachng":"approaching",
"evng":"evening",
"Sumthng":"something",
"EllenPompeo":"Ellen Pompeo",
"shondarhimes":"Shonda Rhimes",
"ABCNetwork":"ABC Network",
"SushmaSwaraj":"Sushma Swaraj",
"pray4japan":"Pray for Japan",
"hope4japan":"Hope for Japan",
"Illusionimagess":"Illusion images",
"SummerUnderTheStars":"Summer Under The Stars",
"ShallWeDance":"Shall We Dance",
"TCMParty":"TCM Party",
"marijuananews":"marijuana news",
"onbeingwithKristaTippett":"on being with Krista Tippett",
"Beingtweets":"Being tweets",
"newauthors":"new authors",
"remedyyyy":"remedy",
"44PM":"44 PM",
"HeadlinesApp":"Headlines App",
"40PM":"40 PM",
"myswc":"Severe Weather Center",
"ithats":"that is",
"icouldsitinthismomentforever":"I could sit in this moment forever",
"FatLoss":"Fat Loss",
"02PM":"02 PM",
"MetroFmTalk":"Metro Fm Talk",
"Bstrd":"bastard",
"bldy":"bloody",
"MetrofmTalk":"Metro Fm Talk",
"terrorismturn":"terrorism turn",
"BBCNewsAsia":"BBC News Asia",
"BehindTheScenes":"Behind The Scenes",
"GeorgeTakei":"George Takei",
"WomensWeeklyMag":"Womens Weekly Magazine",
"SurvivorsGuidetoEarth":"Survivors Guide to Earth",
"incubusband":"incubus band",
"Babypicturethis":"Baby picture this",
"BombEffects":"Bomb Effects",
"win10":"Windows 10",
"idkidk":"I do not know I do not know",
"TheWalkingDead":"The Walking Dead",
"amyschumer":"Amy Schumer",
"crewlist":"crew list",
"Erdogans":"Erdogan",
"BBCLive":"BBC Live",
"TonyAbbottMHR":"Tony Abbott",
"paulmyerscough":"Paul Myerscough",
"georgegallagher":"George Gallagher",
"JimmieJohnson":"Jimmie Johnson",
"pctool":"pc tool",
"DoingHashtagsRight":"Doing Hashtags Right",
"ThrowbackThursday":"Throwback Thursday",
"SnowBackSunday":"Snowback Sunday",
"LakeEffect":"Lake Effect",
"RTphotographyUK":"Richard Thomas Photography UK",
"BigBang_CBS":"Big Bang CBS",
"writerslife":"writers life",
"NaturalBirth":"Natural Birth",
"UnusualWords":"Unusual Words",
"wizkhalifa":"Wiz Khalifa",
"acreativedc":"a creative DC",
"vscodc":"vsco DC",
"VSCOcam":"vsco camera",
"TheBEACHDC":"The beach DC",
"buildingmuseum":"building museum",
"WorldOil":"World Oil",
"redwedding":"red wedding",
"AmazingRaceCanada":"Amazing Race Canada",
"WakeUpAmerica":"Wake Up America",
"\\Allahuakbar\\":"Allahu Akbar",
"bleased":"blessed",
"nigeriantribune":"Nigerian Tribune",
"HIDEO_KOJIMA_EN":"Hideo Kojima",
"FusionFestival":"Fusion Festival",
"50Mixed":"50 Mixed",
"NoAgenda":"No Agenda",
"WhiteGenocide":"White Genocide",
"dirtylying":"dirty lying",
"SyrianRefugees":"Syrian Refugees",
"changetheworld":"change the world",
"Ebolacase":"Ebola case",
"mcgtech":"mcg technologies",
"withweapons":"with weapons",
"advancedwarfare":"advanced warfare",
"letsFootball":"let us Football",
"LateNiteMix":"late night mix",
"PhilCollinsFeed":"Phil Collins",
"RudyHavenstein":"Rudy Havenstein",
"22PM":"22 PM",
"54am":"54 AM",
"38am":"38 AM",
"OldFolkExplainStuff":"Old Folk Explain Stuff",
"BlacklivesMatter":"Black Lives Matter",
"InsaneLimits":"Insane Limits",
"youcantsitwithus":"you cannot sit with us",
"2k15":"2015",
"TheIran":"Iran",
"JimmyFallon":"Jimmy Fallon",
"AlbertBrooks":"Albert Brooks",
"defense_news":"defense news",
"nuclearrcSA":"Nuclear Risk Control Self Assessment",
"Auspol":"Australia Politics",
"NuclearPower":"Nuclear Power",
"WhiteTerrorism":"White Terrorism",
"truthfrequencyradio":"Truth Frequency Radio",
"ErasureIsNotEquality":"Erasure is not equality",
"ProBonoNews":"Pro Bono News",
"JakartaPost":"Jakarta Post",
"toopainful":"too painful",
"melindahaunton":"Melinda Haunton",
"NoNukes":"No Nukes",
"curryspcworld":"Currys PC World",
"ineedcake":"I need cake",
"blackforestgateau":"black forest gateau",
"BBCOne":"BBC One",
"AlexxPage":"Alex Page",
"jonathanserrie":"Jonathan Serrie",
"SocialJerkBlog":"Social Jerk Blog",
"ChelseaVPeretti":"Chelsea Peretti",
"irongiant":"iron giant",
"RonFunches":"Ron Funches",
"TimCook":"Tim Cook",
"sebastianstanisaliveandwell":"Sebastian Stan is alive and well",
"Madsummer":"Mad summer",
"NowYouKnow":"Now you know",
"concertphotography":"concert photography",
"TomLandry":"Tom Landry",
"showgirldayoff":"show girl day off",
"Yougslavia":"Yugoslavia",
"QuantumDataInformatics":"Quantum Data Informatics",
"FromTheDesk":"From The Desk",
"TheaterTrial":"Theater Trial",
"CatoInstitute":"Cato Institute",
"EmekaGift":"Emeka Gift",
"LetsBe_Rational":"Let us be rational",
"Cynicalreality":"Cynical reality",
"FredOlsenCruise":"Fred Olsen Cruise",
"NotSorry":"not sorry",
"UseYourWords":"use your words",
"WordoftheDay":"word of the day",
"Dictionarycom":"Dictionary.com",
"TheBrooklynLife":"The Brooklyn Life",
"jokethey":"joke they",
"nflweek1picks":"NFL week 1 picks",
"uiseful":"useful",
"JusticeDotOrg":"The American Association for Justice",
"autoaccidents":"auto accidents",
"SteveGursten":"Steve Gursten",
"MichiganAutoLaw":"Michigan Auto Law",
"birdgang":"bird gang",
"nflnetwork":"NFL Network",
"NYDNSports":"NY Daily News Sports",
"RVacchianoNYDN":"Ralph Vacchiano NY Daily News",
"EdmontonEsks":"Edmonton Eskimos",
"david_brelsford":"David Brelsford",
"TOI_India":"The Times of India",
"hegot":"he got",
"SkinsOn9":"Skins on 9",
"sothathappened":"so that happened",
"LCOutOfDoors":"LC Out Of Doors",
"NationFirst":"Nation First",
"IndiaToday":"India Today",
"HLPS":"helps",
"HOSTAGESTHROSW":"hostages throw",
"SNCTIONS":"sanctions",
"BidTime":"Bid Time",
"crunchysensible":"crunchy sensible",
"RandomActsOfRomance":"Random acts of romance",
"MomentsAtHill":"Moments at hill",
"eatshit":"eat shit",
"liveleakfun":"live leak fun",
"SahelNews":"Sahel News",
"abc7newsbayarea":"ABC 7 News Bay Area",
"facilitiesmanagement":"facilities management",
"facilitydude":"facility dude",
"CampLogistics":"Camp logistics",
"alaskapublic":"Alaska public",
"MarketResearch":"Market Research",
"AccuracyEsports":"Accuracy Esports",
"TheBodyShopAust":"The Body Shop Australia",
"yychail":"Calgary hail",
"yyctraffic":"Calgary traffic",
"eliotschool":"eliot school",
"TheBrokenCity":"The Broken City",
"OldsFireDept":"Olds Fire Department",
"RiverComplex":"River Complex",
"fieldworksmells":"field work smells",
"IranElection":"Iran Election",
"glowng":"glowing",
"kindlng":"kindling",
"riggd":"rigged",
"slownewsday":"slow news day",
"MyanmarFlood":"Myanmar Flood",
"abc7chicago":"ABC 7 Chicago",
"copolitics":"Colorado Politics",
"AdilGhumro":"Adil Ghumro",
"netbots":"net bots",
"byebyeroad":"bye bye road",
"massiveflooding":"massive flooding",
"EndofUS":"End of United States",
"35PM":"35 PM",
"greektheatrela":"Greek Theatre Los Angeles",
"76mins":"76 minutes",
"publicsafetyfirst":"public safety first",
"livesmatter":"lives matter",
"myhometown":"my hometown",
"tankerfire":"tanker fire",
"MEMORIALDAY":"memorial day",
"MEMORIAL_DAY":"memorial day",
"instaxbooty":"instagram booty",
"Jerusalem_Post":"Jerusalem Post",
"WayneRooney_INA":"Wayne Rooney",
"VirtualReality":"Virtual Reality",
"OculusRift":"Oculus Rift",
"OwenJones84":"Owen Jones",
"jeremycorbyn":"Jeremy Corbyn",
"paulrogers002":"Paul Rogers",
"mortalkombatx":"Mortal Kombat X",
"mortalkombat":"Mortal Kombat",
"FilipeCoelho92":"Filipe Coelho",
"OnlyQuakeNews":"Only Quake News",
"kostumes":"costumes",
"YEEESSSS":"yes",
"ToshikazuKatayama":"Toshikazu Katayama",
"IntlDevelopment":"Intl Development",
"ExtremeWeather":"Extreme Weather",
"WereNotGruberVoters":"We are not gruber voters",
"NewsThousands":"News Thousands",
"EdmundAdamus":"Edmund Adamus",
"EyewitnessWV":"Eye witness WV",
"PhiladelphiaMuseu":"Philadelphia Museum",
"DublinComicCon":"Dublin Comic Con",
"NicholasBrendon":"Nicholas Brendon",
"Alltheway80s":"All the way 80s",
"FromTheField":"From the field",
"NorthIowa":"North Iowa",
"WillowFire":"Willow Fire",
"MadRiverComplex":"Mad River Complex",
"feelingmanly":"feeling manly",
"stillnotoverit":"still not over it",
"FortitudeValley":"Fortitude Valley",
"CoastpowerlineTramTr":"Coast powerline",
"ServicesGold":"Services Gold",
"NewsbrokenEmergency":"News broken emergency",
"Evaucation":"evacuation",
"leaveevacuateexitbe":"leave evacuate exit be",
"P_EOPLE":"PEOPLE",
"Tubestrike":"tube strike",
"CLASS_SICK":"CLASS SICK",
"localplumber":"local plumber",
"awesomejobsiri":"awesome job siri",
"PayForItHow":"Pay for it how",
"ThisIsAfrica":"This is Africa",
"crimeairnetwork":"crime air network",
"KimAcheson":"Kim Acheson",
"cityofcalgary":"City of Calgary",
"prosyndicate":"pro syndicate",
"660NEWS":"660 NEWS",
"BusInsMagazine":"Business Insurance Magazine",
"wfocus":"focus",
"ShastaDam":"Shasta Dam",
"go2MarkFranco":"Mark Franco",
"StephGHinojosa":"Steph Hinojosa",
"Nashgrier":"Nash Grier",
"NashNewVideo":"Nash new video",
"IWouldntGetElectedBecause":"I would not get elected because",
"SHGames":"Sledgehammer Games",
"bedhair":"bed hair",
"JoelHeyman":"Joel Heyman",
"viaYouTube":"via YouTube",
}

def clean_non_latin(text, lang):
    if lang != 'ru':
        return re.sub('[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', '', text)
    else:
        return text

def replace_invalid_words(text, lang, dict_type):
    if lang == 'en' or dict_type in ['special_chars', 'character_entity', 'hashtag']:
        if dict_type == 'special_chars':
            dict_ = twitter_special_chars
        elif dict_type == 'character_entity':
            dict_ = twitter_character_entity
        elif dict_type == 'typos_slang':
            dict_ = twitter_typos_slang
        elif dict_type == 'hashtag':
            dict_ = twitter_hashtag
        else:
            raise ValueError('Invalid dict type given')
        
        mispellings, mispellings_re = _get_mispell(dict_)
    
        def replace(match):
            return mispellings[match.group(0)]
    
        return mispellings_re.sub(replace, text)
    else:
        return text


emoji_pattern = re.compile("["
                   u"\U0001F600-\U0001F64F"  # emoticons
                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                   u"\U00002702-\U000027B0"
                   u"\U000024C2-\U0001F251"
                   "]+", flags=re.UNICODE)
def remove_invalid_char(string):
    return emoji_pattern.sub(r'', string)

def remove_emoji(text):
    text = remove_invalid_char(text)
    return emoji.get_emoji_regexp().sub(u'', text)

def clean_twit(df, columns: list):
    df_new = df.copy()
    for col in columns:
        if col in df_new.columns:
            # df_new[col] = df_new[col].apply(lambda x: clean_numbers(x))
            df_new[col] = df_new[[col, 'lang']].parallel_apply(lambda x: clean_text(x[col], x['lang']), axis=1)
#             df_new[col] = df_new[col].parallel_apply(lambda x: clean_puncts(x.lower())) 
            df_new[col] = df_new[col].apply(lambda x: remove_emoji(x.lower())) 
            df_new[col] = df_new[[col, 'lang']].parallel_apply(lambda x: replace_invalid_words(x[col], x['lang'], 'special_chars'), axis=1)
            df_new[col] = df_new[[col, 'lang']].parallel_apply(lambda x: replace_invalid_words(x[col], x['lang'], 'character_entity'), axis=1)
            df_new[col] = df_new[[col, 'lang']].parallel_apply(lambda x: replace_invalid_words(x[col], x['lang'], 'typos_slang'), axis=1)
            df_new[col] = df_new[[col, 'lang']].parallel_apply(lambda x: replace_invalid_words(x[col], x['lang'], 'hashtag'), axis=1)
            df_new[col] = df_new[[col, 'lang']].parallel_apply(lambda x: clean_non_latin(x[col], x['lang']), axis=1)
            # df_new[col] = df_new[[col, 'lang']].parallel_apply(lambda x: handle_contractions(x[col], x['lang']), axis=1)  
#             df_new[col] = df_new[col].parallel_apply(lambda x: fix_quote(x))   
    
    return df_new

