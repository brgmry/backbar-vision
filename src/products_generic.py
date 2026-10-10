"""Generic open-vocabulary label list for the zero-shot crop classifier.

Written from general knowledge of what sits on a UK bar back or kitchen shelf: the best-selling
spirit and liqueur brands worldwide, common mixers and soft drinks, and kitchen staples by name.
Not derived from data/labelled/ or from looking at the photos. Rows: brand, product, category, liquid, size_ml.
"""

SPIRITS = [
    ("Smirnoff", "No. 21 Vodka", "vodka"), ("Absolut", "Vodka", "vodka"), ("Grey Goose", "Vodka", "vodka"),
    ("Belvedere", "Vodka", "vodka"), ("Ketel One", "Vodka", "vodka"), ("Russian Standard", "Vodka", "vodka"),
    ("Au", "Vodka", "vodka"), ("Ciroc", "Vodka", "vodka"), ("Tito's", "Handmade Vodka", "vodka"),
    ("Gordon's", "London Dry Gin", "gin"), ("Gordon's", "Pink Gin", "gin"), ("Tanqueray", "London Dry Gin", "gin"),
    ("Bombay Sapphire", "Gin", "gin"), ("Hendrick's", "Gin", "gin"), ("Beefeater", "London Dry Gin", "gin"),
    ("Whitley Neill", "Rhubarb & Ginger Gin", "gin"), ("Roku", "Gin", "gin"), ("Malfy", "Gin Rosa", "gin"),
    ("Jack Daniel's", "Old No. 7 Tennessee Whiskey", "whiskey"), ("Jameson", "Irish Whiskey", "whiskey"),
    ("Johnnie Walker", "Red Label", "whisky"), ("Johnnie Walker", "Black Label", "whisky"),
    ("Famous Grouse", "Blended Scotch Whisky", "whisky"), ("Glenfiddich", "12 Year Old", "whisky"),
    ("Jim Beam", "Bourbon", "whiskey"), ("Maker's Mark", "Bourbon", "whiskey"), ("Monkey Shoulder", "Blended Malt", "whisky"),
    ("Chivas Regal", "12 Year Old", "whisky"), ("Bell's", "Original Whisky", "whisky"), ("Glenmorangie", "Original", "whisky"),
    ("Bacardi", "Carta Blanca Rum", "rum"), ("Captain Morgan", "Original Spiced Gold", "rum"),
    ("Kraken", "Black Spiced Rum", "rum"), ("Havana Club", "3 Anos", "rum"), ("Sailor Jerry", "Spiced Rum", "rum"),
    ("Malibu", "Coconut Rum", "rum"), ("Lamb's", "Navy Rum", "rum"), ("Diplomatico", "Reserva Exclusiva", "rum"),
    ("Jose Cuervo", "Especial Tequila", "tequila"), ("Patron", "Silver Tequila", "tequila"),
    ("Don Julio", "Blanco Tequila", "tequila"), ("Olmeca", "Tequila", "tequila"),
    ("Hennessy", "VS Cognac", "cognac"), ("Courvoisier", "VS Cognac", "cognac"), ("Remy Martin", "VSOP Cognac", "cognac"),
]
LIQUEURS = [
    ("Baileys", "Original Irish Cream", "cream liqueur"), ("Kahlua", "Coffee Liqueur", "coffee liqueur"),
    ("Tia Maria", "Coffee Liqueur", "coffee liqueur"), ("Cointreau", "Orange Liqueur", "orange liqueur"),
    ("Aperol", "Aperitivo", "aperitif"), ("Campari", "Bitter", "bitter"), ("Jagermeister", "Herbal Liqueur", "herbal liqueur"),
    ("Disaronno", "Amaretto", "amaretto"), ("Martini", "Rosso Vermouth", "vermouth"), ("Martini", "Extra Dry Vermouth", "vermouth"),
    ("Southern Comfort", "Liqueur", "liqueur"), ("Sambuca", "Molinari", "sambuca"), ("Chambord", "Raspberry Liqueur", "liqueur"),
    ("Midori", "Melon Liqueur", "liqueur"), ("Archers", "Peach Schnapps", "schnapps"), ("Pimm's", "No. 1 Cup", "liqueur"),
    ("Licor 43", "Liqueur", "liqueur"), ("St-Germain", "Elderflower Liqueur", "liqueur"), ("Angostura", "Aromatic Bitters", "bitters"),
]
MIXERS = [
    ("Schweppes", "Indian Tonic Water", "mixer", "tonic water"), ("Fever-Tree", "Premium Indian Tonic Water", "mixer", "tonic water"),
    ("Fever-Tree", "Ginger Beer", "mixer", "ginger beer"), ("Schweppes", "Soda Water", "mixer", "soda water"),
    ("Coca-Cola", "Original Taste", "soft_drink", "cola"), ("Coca-Cola", "Zero Sugar", "soft_drink", "cola"),
    ("Diet Coke", "Cola", "soft_drink", "cola"), ("Pepsi", "Max", "soft_drink", "cola"), ("Sprite", "Lemon Lime", "soft_drink", "lemonade"),
    ("7Up", "Lemon Lime", "soft_drink", "lemonade"), ("Fanta", "Orange", "soft_drink", "orange soda"), ("Irn-Bru", "Original", "soft_drink", "soda"),
    ("Red Bull", "Energy Drink", "soft_drink", "energy drink"), ("Robinsons", "Orange Squash", "soft_drink", "squash"),
    ("Ribena", "Blackcurrant", "soft_drink", "cordial"), ("Rose's", "Lime Juice Cordial", "mixer", "lime cordial"),
    ("Belvoir", "Elderflower Cordial", "soft_drink", "cordial"), ("Tropicana", "Orange Juice", "soft_drink", "orange juice"),
    ("Innocent", "Orange Juice", "soft_drink", "orange juice"), ("Ocean Spray", "Cranberry Juice", "mixer", "cranberry juice"),
    ("Evian", "Natural Mineral Water", "soft_drink", "water"), ("Volvic", "Natural Mineral Water", "soft_drink", "water"),
    ("Monin", "Sugar Syrup", "syrup", "sugar syrup"), ("Monin", "Vanilla Syrup", "syrup", "vanilla syrup"),
    ("Monin", "Grenadine Syrup", "syrup", "grenadine"), ("Monin", "Caramel Syrup", "syrup", "caramel syrup"),
    ("Lyle's", "Golden Syrup", "syrup", "golden syrup"), ("Sweet Freedom", "Maple Syrup", "syrup", "maple syrup"),
]
BEER_WINE = [
    ("Heineken", "Lager", "beer", "lager"), ("Stella Artois", "Lager", "beer", "lager"), ("Peroni", "Nastro Azzurro", "beer", "lager"),
    ("Corona", "Extra", "beer", "lager"), ("Guinness", "Draught Stout", "beer", "stout"), ("Budweiser", "Lager", "beer", "lager"),
    ("Birra Moretti", "Lager", "beer", "lager"), ("Asahi", "Super Dry", "beer", "lager"), ("Kopparberg", "Cider", "beer", "cider"),
    ("Echo Falls", "Red Wine", "wine", "red wine"), ("Yellow Tail", "Shiraz", "wine", "red wine"), ("Jacob's Creek", "Chardonnay", "wine", "white wine"),
    ("Casillero del Diablo", "Cabernet Sauvignon", "wine", "red wine"), ("Oyster Bay", "Sauvignon Blanc", "wine", "white wine"),
    ("Prosecco", "Sparkling Wine", "wine", "sparkling wine"), ("Moet & Chandon", "Champagne", "wine", "champagne"),
    ("Blossom Hill", "Rose Wine", "wine", "rose wine"),
]
KITCHEN = [
    ("Heinz", "Tomato Ketchup", "sauce", "tomato ketchup"), ("", "Tomato Ketchup", "sauce", "tomato ketchup"),
    ("Hellmann's", "Real Mayonnaise", "sauce", "mayonnaise"), ("", "Mayonnaise", "sauce", "mayonnaise"),
    ("HP", "Brown Sauce", "sauce", "brown sauce"), ("Lea & Perrins", "Worcestershire Sauce", "sauce", "worcestershire sauce"),
    ("Tabasco", "Pepper Sauce", "sauce", "hot sauce"), ("Huy Fong", "Sriracha", "sauce", "chilli sauce"),
    ("Frank's", "RedHot Sauce", "sauce", "hot sauce"), ("Nando's", "Peri-Peri Sauce", "sauce", "hot sauce"),
    ("", "Sweet Chilli Sauce", "sauce", "sweet chilli sauce"), ("Kikkoman", "Soy Sauce", "sauce", "soy sauce"),
    ("Blue Dragon", "Soy Sauce", "sauce", "soy sauce"), ("", "Fish Sauce", "sauce", "fish sauce"), ("", "Oyster Sauce", "sauce", "oyster sauce"),
    ("Colman's", "English Mustard", "sauce", "mustard"), ("", "Barbecue Sauce", "sauce", "bbq sauce"), ("", "Salad Cream", "sauce", "salad cream"),
    ("Filippo Berio", "Extra Virgin Olive Oil", "oil", "olive oil"), ("", "Extra Virgin Olive Oil", "oil", "olive oil"),
    ("", "Vegetable Oil", "oil", "vegetable oil"), ("", "Sunflower Oil", "oil", "sunflower oil"), ("", "Rapeseed Oil", "oil", "rapeseed oil"),
    ("", "Sesame Oil", "oil", "sesame oil"), ("", "Balsamic Vinegar", "other", "vinegar"), ("Sarson's", "Malt Vinegar", "other", "vinegar"),
    ("", "White Wine Vinegar", "other", "vinegar"), ("", "Lemon Juice", "other", "lemon juice"), ("", "Lime Juice", "other", "lime juice"),
    ("", "Whole Milk", "dairy", "whole milk"), ("", "Semi Skimmed Milk", "dairy", "semi skimmed milk"), ("", "Skimmed Milk", "dairy", "skimmed milk"),
    ("Cravendale", "Milk", "dairy", "milk"), ("Arla", "Lactofree Milk", "dairy", "milk"), ("Alpro", "Oat Drink", "dairy", "oat milk"),
    ("Oatly", "Oat Drink Barista", "dairy", "oat milk"), ("", "Double Cream", "dairy", "cream"), ("", "Single Cream", "dairy", "cream"),
    ("", "Honey", "syrup", "honey"), ("", "Agave Syrup", "syrup", "agave syrup"), ("", "Chocolate Sauce", "sauce", "chocolate sauce"),
    ("Fairy", "Washing Up Liquid", "other", "washing up liquid"),
]
SIZE = {"spirit": 700, "liqueur": 700, "mixer": 500, "soft_drink": 500, "syrup": 700, "wine": 750, "beer": 330,
        "sauce": 500, "oil": 500, "dairy": 1136, "other": 500}

# generic things that are not stock; a crop closest to one of these is dropped
CLUTTER = ["a reusable water bottle", "a cocktail shaker", "a mug", "a drinking glass", "a roll of kitchen paper",
           "a plastic cup", "a jar lid", "a person's hand", "an empty table"]


def products() -> list[dict]:
    rows = [dict(brand=b, product=p, category="spirit", liquid=l) for b, p, l in SPIRITS]
    rows += [dict(brand=b, product=p, category="liqueur", liquid=l) for b, p, l in LIQUEURS]
    rows += [dict(brand=b, product=p, category=c, liquid=l) for b, p, c, l in MIXERS + BEER_WINE + KITCHEN]
    for r in rows:
        r["size"] = SIZE[r["category"]]
    return rows
