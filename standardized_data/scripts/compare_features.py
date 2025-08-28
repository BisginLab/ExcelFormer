import json
HARD_CODED = [
  'ContentRating','Genre','CurrentVersion','AndroidVersion','DeveloperCategory',
  'lowest_android_version','highest_android_version','privacy_policy_link',
  'developer_website','days_since_last_update','isSpamming','max_downloads_log',
  'LenWhatsNew','PHONE','OneStarRatings','developer_address','FourStarRatings',
  'intent','ReviewsAverage','STORAGE','LastUpdated','TwoStarRatings',
  'LOCATION','FiveStarRatings','ThreeStarRatings'
]

meta = json.load(open("standardized_data/meta/features_v1.json"))
mi25 = meta["canonical_feature_order"]

print("Equal order? ", HARD_CODED == mi25)
print("Set equal?   ", set(HARD_CODED) == set(mi25))
if HARD_CODED != mi25:
    print("\nFirst diff by position:")
    for i,(a,b) in enumerate(zip(HARD_CODED, mi25)):
        if a!=b: print(i, a, "vs", b)
