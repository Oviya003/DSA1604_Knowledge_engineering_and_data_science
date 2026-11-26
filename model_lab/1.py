
from owlready2 import *

onto = get_ontology("http://travel.org/onto.owl")

with onto:
    class Interest(Thing):
        pass

    class AdventureInterest(Interest):
        pass

    class CulturalInterest(Interest):
        pass

    class WaterSport(AdventureInterest):
        pass

    class ScubaDiving(WaterSport):
        pass

    class Traveller(Thing):
        pass

    class TourPackage(Thing):
        pass

    class Activity(Thing):
        pass

    class hasInterest(ObjectProperty):
        domain = [Traveller]
        range = [Interest]

    class offersActivity(ObjectProperty):
        domain = [TourPackage]
        range = [Activity]

    class hasDestination(DataProperty):
        domain = [TourPackage]
        range = [str]

    class interestLevel(DataProperty):
        domain = [Traveller]
        range = [str]

    # Create objects
    t1 = Traveller("Traveller1")
    p1 = TourPackage("Package1")
    a1 = ScubaDiving("Scuba")

    t1.hasInterest = [a1]
    p1.offersActivity = [a1]
    p1.hasDestination = ["Goa"]
    t1.interestLevel = ["High"]

# Synonym handling
synonyms = {
    "trekking": "hiking",
    "hiking": "hiking"
}

def normalize(word):
    return synonyms.get(word.lower(), word.lower())

onto.save(file="travel.owl")

print("Ontology created successfully")
print("Traveller:", t1.name)
print("Interest:", a1.name)
print("Destination:", p1.hasDestination)
print("Trekking:", normalize("trekking"))
