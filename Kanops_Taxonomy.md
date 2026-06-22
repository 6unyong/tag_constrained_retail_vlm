# Kanops Classification Taxonomy

**The taxonomy used by Argus and the other classifiers.**

Source of truth: `lib/argus_taxonomy.py` (the "One Brain" — signed off 6 Jan 2026, spatial axes revised 6/9 Feb 2026).
Per-model label spaces: `training/config/models.yaml` + `training/config/mercator_category_mapping.json`.
Generated 17 Jun 2026 — values transcribed verbatim from the live modules.

---

## 1. How the taxonomy is structured

Argus classifies every archive image on **three independent axes**:

| Axis | Question | Source | Required |
|------|----------|--------|----------|
| **Category** | *What* product? | `TAXONOMY` (21 parents → 227 subcategories) | Yes |
| **Location** | *Where* in store? | `LOCATIONS` (13) | Yes |
| **Fixture** | *On what* display? | `FIXTURES` (8) | Optional — tag when obvious |

Design principle (verbatim from source): **"Would a client pay differently for A vs B?"** If not, they are the same category. A fixture can appear at *any* location (a Shipper at Checkout, at Foyer, in Seasonal Space).

The master taxonomy is the **single normalisation target**. Each classifier emits its own coarser label set, then mappings fold those back into the master parents.

---

## 2. Master Product Taxonomy — 21 parents · 227 subcategories

### Food — Ambient

**Ambient Grocery** *(24)* — shelf-stable food products
> Ambient Desserts and Puddings · Baking Ingredients · Biscuits · Biscuits for Cheese · Canned Fish · Canned Goods · Cereal Bars and Snacking · Cereals · Confectionery · Cooking Ingredients and Sauces · Crisps, Nuts and Snacks · Energy Drinks · Packet and Instant Snacks · Pasta, Rice and Grains · Pickles and Condiments · Plant-Based Drinks · Ready to Drink · Sauces and Ketchups · Soft Drinks and Mixers · Soup · Spreads and Preserves · Stuffing, Gravies and Accompaniments · Tea and Coffee · World Foods

### Food — Chilled

**Chilled and Dairy** *(15)* — refrigerated food products
> Butter and Spreads · Cheese · Chilled Accompaniments · Chilled Party Foods · Cooked Meats · Deli Counter · Dips and Deli · Eggs · Meal Deals · Milk and Cream · Milk Alternatives · Ready Meals · Sandwiches and Wraps · Vegetarian and Vegan · Yoghurts and Desserts

### Food — Fresh Meat

**Fresh Meat and Poultry** *(9)* — fresh uncooked meat products
> BBQ · Beef · Lamb · Meat Counter · Mince · Other Poultry and Game · Pork · Poultry (inc Turkey) · Sausages and Burgers

### Food — Fresh Fish

**Fresh Fish and Seafood** *(3)* — fresh fish and seafood products
> Fish · Fish Counter · Prawns and Shellfish

### Food — Fresh Produce

**Fresh Produce** *(11)* — fresh fruit, vegetables, salads
> Exotic Fruit · Fresh Herbs · Dried Fruit and Nuts · Fruit · Mushrooms · Potatoes · Prepared Fruit · Prepared Salads · Prepared Vegetables · Salad · Vegetables

### Food — Bakery

**Bakery** *(11)* — fresh and packaged bakery products
> Bakery Counter · Bread and Rolls · Cakes and Treats · Celebration Cakes · Doughnuts · Free From · Fresh Croissants and Pastries · Mini Bites · Morning Goods · Muffins and Pancakes · Slices, Rolls and Logs

### Food — Frozen

**Frozen** *(12)* — frozen food products
> Frozen Chips and Potatoes · Frozen Desserts · Frozen Fish · Frozen Free From · Frozen Fruit and Veg · Frozen Meat · Frozen Meat Free · Frozen Party Food · Frozen Pizza · Frozen Ready Meals · Frozen World Foods · Ice Cream

### Drinks — Alcohol

**Beers, Wines and Spirits** *(10)* — alcoholic beverages
> Beer and Cider · Champagne and Sparkling · Fortified Wine · Liqueurs and Cream Liqueurs · Low and No Alcohol · Pre-mixed Cocktails · Red Wine · Rose Wine · Spirits · White Wine

### Non-Food — Health & Beauty

**Health and Beauty** *(12)* — personal care and beauty products
> Bathing and Shower · Cosmetics · Deodorant · Feminine Care · Fragrance · Haircare · Hand Washes · Male Grooming · Medicine and Pharmacy · Oral Care · Skincare · Sun Care

### Non-Food — Household

**Household and Pet** *(12)* — cleaning, laundry, pet products
> Air Care · Bin Bags · Cleaning Products · Dishwashing · Foils and Wraps · Fuel and Firewood · Kitchen Roll and Tissues · Laundry · Pet Food · Pet Treats and Accessories · Shoe Care · Toilet Roll

### Non-Food — Baby

**Baby and Infant** *(4)* — baby food, formula, nappies, accessories
> Baby Equipment · Baby Food · Baby Formula · Baby Supplies

### Non-Food — General Merchandise — Gifting

**GM Gifting** *(13)* — gift items and accessories
> Alcohol Gifts · Beauty Gifts · Books · Calendars and Diaries · Electronics Gifts · Experience Vouchers · Fashion Gifts · Food Gifts · Fragrance Gifts · Gift Cards · Hampers · Pet Gifts · Stationery Gifts

### Non-Food — General Merchandise — Homeware

**GM Homeware** *(10)* — home and kitchen products
> Bathroom Accessories · Bedding · Candles and Fragrance · Cook and Bakeware · Cups and Mugs · Homewares · Hosting and Tableware · Lightbulbs · Lunch Bags and Boxes · Storage

### Non-Food — General Merchandise — Electricals

**GM Electricals** *(8)* — electrical and tech products
> Audio and Headphones · Batteries · Gaming · Kitchen Electricals · Personal Care Electricals · Phone Accessories · Smart Home · TV and Home Entertainment

### Seasonal

**Seasonal** *(17)* — seasonal and event-specific products
> Advent Calendars · Cards and Wrap · Christmas Crackers · Christmas Decorations · Christmas Lights · Christmas Trees · Easter Eggs · Halloween · Seasonal Bakery · Seasonal Biscuits · Seasonal Chocolates · Seasonal Non-Food · Selection Boxes · Sweets and Tubs · Trick or Treat · Valentines · Wreaths and Garlands
>
> *Note: Seasonal Bakery includes Mince Pies, Stollen, Hot Cross Buns, Christmas Cakes.*

### Clothing

**Clothing** *(12)* — apparel and accessories
> Baby Clothing · Charity · Childrens Clothing · Hair Accessories · Jewellery · Loungewear and Pyjamas · Menswear · Novelty and Festive · School Uniform · Socks and Slippers · Underwear · Womenswear

### Entertainment

**Toys and Entertainment** *(7)* — toys, games, media
> Arts and Crafts · Books and Annuals · Collectibles · DVDs and CDs · Games and Puzzles · Outdoor Toys · Toys

### Store Environment *(non-product)*

**Store Environment** *(21)* — store fixtures, signage, environment
> Aisle Ends · Baskets and Trolleys · Branding and Signage · Checkouts · Clip Strips · Dump Bins · Empty Shelf · Feature Space · Foyer · Gondola Ends · Ladder Rack · Pallet Displays · Power Aisle · Promotional Displays · Queuing Systems · Reduction Area · Sendai Units · Shelf Edge · Shippers and FSUs · Stacks · Staff

### Special / Dietary

**Special Dietary** *(5)* — free from, vegan, organic products
> Free From · Halal · Kosher · Organic · Vegan

### Floral

**Floral** *(6)* — plants, flowers, bouquets
> Bouquets · Daffodils · Flowers · Indoor Plants · Outdoor Plants · Poinsettias

### Automotive and DIY

**Automotive and DIY** *(5)* — car care, tools, DIY, garden products
> Car Accessories · Car Care · DIY · Garden · Stones and Chips

> Plus one edge-case bucket: **`Review`** — review folder for items that can't be confidently placed.

---

## 3. Spatial axes

### Locations (13) — *where in store*

| Location | Description |
|----------|-------------|
| Aisle | Standard gondola run, regular shelf space |
| Central Aisle | Main walkway through store centre, high-traffic promotional space |
| Back Aisle | Rear of store aisle, typically staples and bulk items |
| Gondola End | End of aisle, end cap — premium promotional space |
| Seasonal Space | Dedicated seasonal or promotional area (not gondola end) |
| Foyer | Store entrance area, front of store, window displays |
| Checkout | Till area, queuing lanes, self-service — impulse zone |
| Fresh Counter | Staffed service counters (deli, bakery, fish, meat, cheese) |
| Chiller | Refrigerated cabinets and chilled sections |
| Freezer | Frozen food cabinets and sections |
| Produce | Fresh produce area — fruit, veg, salad, herbs |
| External | Outside the store — car park, forecourt, store front, signage |
| Unknown | Location cannot be determined from image |

### Fixtures (8) — *on what display*

| Fixture | Description |
|---------|-------------|
| Shelf | Standard gondola shelving, fixed racking |
| Floor Stack | Product stacked on pallet or plinth on the floor |
| Shipper | Free-standing display unit, branded cardboard display, FSDU |
| Dump Bin | Open bin display, wire or cardboard |
| Hanging Display | Clip strips, peg hooks, J-hooks, pegboard displays |
| Baskets | Wire baskets, wooden crates, open produce-style displays |
| Display Unit | Non-branded display stand, counter unit, or racking not fitting other categories |
| Unknown | Fixture type cannot be determined from image |

### Auto-linking hints (save-a-click defaults)

When a category can *only* appear in one location, it's auto-set:

- **Parent → location:** Chilled and Dairy → Chiller · Fresh Meat and Poultry → Chiller · Fresh Fish and Seafood → Chiller · Fresh Produce → Produce · Frozen → Freezer · Floral → Produce
- **Subcategory → location (overrides parent):** Deli Counter / Fish Counter / Meat Counter / Bakery Counter → Fresh Counter · Ice Cream → Freezer

---

## 4. Per-model label spaces

Each classifier emits its own (usually coarser) set, normalised back into the master taxonomy via the mappings in §5.

| Model | Codename role | Type | Classes |
|-------|---------------|------|---------|
| **Argus** | Full-archive classifier (:8508) | Multi-axis classification | Full master taxonomy: 21 parents / 227 subcategories + 13 locations + 8 fixtures |
| **Mercator** | Category model | Classification | 20 consolidated categories (see below) |
| **Vertumnus** | Seasonal category | Classification | 10 |
| **Chronos** | Seasonal event pre-stage | Classification | 8 |
| **Themis** | Shelf shape detection | Object detection | 8 |
| **Idaten-K / MerKury** | Shipper detection | Object detection | 1 |
| **Shelf-Standards** | On-shelf availability | Object detection | 4 |
| **Pomona** | Per-event subcategory | Classification | Dynamic (loaded per event from `event_subcategories.json`) |

### Mercator — 20 consolidated categories (v5.4, 22 Mar 2026)
Folds the fine-grained Argus taxonomy down to a trainable set of 20:

> Advent Calendars · Ambient Grocery · Bakery · Batteries and Electricals · Beers, Wines and Spirits · Cards and Wrap · Chilled and Dairy · Clothing · Confectionery and Biscuits · Decorations · Easter Eggs · Fruit, Veg and Floral · Garden and Outdoor · Gifting · Health, Beauty and Pharmacy · Homeware · Household · Mince Pies · Snacks · Stationery · Toys, Entertainment and Collectibles

*Excluded from training:* GM Car Care and Accessories · Chinese New Year · Branding and Signage · Shippers · EXCLUDE

### Vertumnus — 10 seasonal categories
> Ambient · Beers, Wines and Spirits · Branding and Signage · Cards and Wrap · Clothing · Fresh Foods and Bakery · Frozen · GM · Produce and Floral · Seasonal Confectionery

### Chronos — 8 seasonal events
> Back to School · Black Friday · Christmas · Easter · Father's Day · Halloween · Mother's Day · Valentine's Day

### Themis — 8 shelf shapes (detection)
> sel · product · srp · promo_signage · flag · gap · signage · shipper

### Idaten-K / MerKury — 1 class (detection)
> shipper

### Shelf-Standards — 4 classes (detection)
> gap · misplaced · damaged · wrong_price

### Pomona — dynamic
Per-event subcategories loaded at runtime from `training/config/event_subcategories.json` (event folder names become the class set).

---

## 5. Normalisation mappings

Defined in `lib/argus_taxonomy.py`, these fold each system's output back to Argus parents:

- **`SATURNALIA_TO_ARGUS`** — seasonal/Christmas classifier names → Argus parent (e.g. `Core Ambient Grocery` → `Ambient Grocery`). Built from each parent's `saturnalia_aliases`.
- **`SHIPPER_TO_ARGUS`** — shipper-detection parent → Argus parent (e.g. `Household` → `Household and Pet`). Built from each parent's `shipper_parent`.
- **`CHRISTMAS_TO_ARGUS`** — backward-compat alias for `SATURNALIA_TO_ARGUS`.
- **`mercator_category_mapping.json`** — fine-grained → 20 Mercator categories.

### Utility functions (`lib/argus_taxonomy.py`)
```python
get_parent(category)            # parent for any subcategory
get_subcategories(parent)       # subcategories under a parent
normalize_category(cat, source) # normalise from any system into Argus
get_all_categories()            # flat list of all 248 (21 + 227)
get_all_parents()               # the 21 parents
validate_category(category)     # (bool, reason)
get_location_hints(parent, sub) # auto-link location/fixture
stats() / spatial_stats()       # taxonomy statistics
```

---

## 6. File reference

| Purpose | Path |
|---------|------|
| Master taxonomy + locations/fixtures + mappings + utils | `lib/argus_taxonomy.py` |
| Per-model class definitions | `training/config/models.yaml` |
| Mercator 20-category consolidation | `training/config/mercator_category_mapping.json` |
| Pomona per-event subcategories | `training/config/event_subcategories.json` |
| Argus classifier UI | `argus/classifier_gui.py` (:8508) |
| Legacy annotation categories | `lib/categories.py` |
