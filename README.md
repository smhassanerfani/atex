# ATeX - [ATLANTIS](https://github.com/smhassanerfani/atlantis) TeXture Dataset
This is the respository for the ATex Dataset.

## Overview
A benchmark for image textures analysis of water in different waterbodies. This texture analysis of water in different waterbodies contributes to the [ATLANTIS](https://github.com/smhassanerfani/atlantis) dataset.

## Dataset Statistics
The current version of this dataset contains:
* Polygon annotations with attributes, annotation times, and foreground/background ordering
* ATeX patches use **32 x 32** pixels within **15** distinct waterbodies
* A total of **12,503** patches 
    - Training: **8,753**
    - Validation: **1,252**
    - Testing: **2,498**
* **35** labels

|Labels|Description|
|---|---|
|Estuary|A partially enclosed waterbody that is always near the coastline, except for "freshwater estuaries created when a river flows into a freshwater lake". As freshwater mixes with saltwater it creates brackish water which is commonly found in Estuaries. A variety of estuary types exist, with “1) coastal plain (drowned river valley) estuaries; 2) tectonic estuaries; 3) bar-built estuaries; and 4) fjord estuaries,” which are created by “sea level rise and filling in an existing river valley, tectonic activity, lagoon or bay protection by a sandbar or barrier island, and glaciers,” respectively ([National Geographic Society](https://www.nationalgeographic.org/encyclopedia/estuary/)).|
|River Delta|Occurs as a “wetland that forms as rivers empty their water and sediment into another body of water, such as an ocean, lake, or another river,” and may “also empty into land,” though this is rare. The rivers that flow into deltas are usually calm and slow. They are usually shaped in a triangle, with there being more outlets than inlets ([National Geographic Society](https://www.nationalgeographic.org/encyclopedia/delta/)).|
|Pool|Pools are often enclosed with a smooth, well defined borders, contain still or splashing water, and are clear from treatment. The ‘[Better Health Channel](https://www.betterhealth.vic.gov.au/health/HealthyLiving/swimming-pools-water-quality)’ states that features of a healthy pool include water that is “clear,” with the ability to “see the bottom of the pool”.|
|Hot Spring|"Forms when water deep below the Earth’s surface is heated by rocks or other means, and rises to the Earth’s surface". ([World of Phenomena](www.phenomena.org/geological/hotspring/)). Hot spring water temperature averages at “143 degrees Fahrenheit,” with “the temperature of earth’s crust increas[ing] with depth, average—3 to 5 degrees F for every 300 feet down” ([National Park Service](https://www.nps.gov/yell/learn/nature/hot-springs.htm)). However, a more general definition of a hot springs’ temperature has been said to be anything over 98 degrees Fahrenheit ([National Park Service](https://www.nps.gov/hosp/learn/education/upload/followthewater_final.pdf)).|
|Glacier|"Ice masses that move under their own weight, and currently cover ~ 10% of the land surface of the Earth” ([Copland](https://doi.org/10.1016/B978-0-12-818234-5.00014-6)). Glaciers fall into 2 catagories: Alpine and Ice sheets. Alpine glaciers are pulled by gravity and move slowly through a valley, while ice sheets spread out from its center and the ice acts like a liquid in the way it covers the ground ([National Geographic Society](https://www.nationalgeographic.org/encyclopedia/glacier/)).|
|Waterfall|Also known as cascades, are notable for being “a river or other body of water’s steep fall over a rocky ledge into a plunge pool below,” where the plunge pool is just a landing spot forthe cascading water. Waterfalls form in a variety of different ways, with multiple visual features resulting from how they formed, but they “often form as streams flow from soft rock to hard rock. The soft rock erodes, leaving a hard ledge over which the stream falls such as granite formations forming cliffs and ledges” from the “stream’s channel cutting so deep into the stream bed” ([National Geographic Society](www.nationalgeographic.org/encyclopedia/waterfall/)).|
|Lake|Relatively calm or still water with little to no blurring of reflections of the surrounding vegetation,terrain, or sky, and the waterbody being either opaque or transparent while also being generally clear of most surface vegetation that is common in other waterbodies like wetlands. There is oftentimes no visible direction for the water’s flow, except for potentially that which is caused by wind.|
|Wetland|“an area of land either covered by water or saturated with water,” is “neither totally dry land nor totally underwear; they have characteristics of both” ([National Geographic Society](https://www.nationalgeographic.org/encyclopedia/wetland/)) Wetlands can be identified by the herbaceous species that populate it, such as the thin, emergent or floating vegetation that will appear to be stranded within the water ([Wildfowl & Wetlands Trust Limited](https://www.wwt.org.uk/discover-wetlands/wetlands/wetland-habitats/#:~:text=Technically%2C%20wetlands%20are%20unique%20ecosystems,over%20time%20into%20different%20forms)).|
|Rapids|Turbulence of water and the white foam created by this turbulence, the water appearing to be moving quite quickly, and other areas close to these ‘rapids’ that appear more still. Another feature common in rapids are large, strong rocks that are present in the river or stream, emerging from the waterbody’s surface. These emerging rocks cause the flow to change directions and crossing flow directions and areas of swirling water help us recognize rapids.|
|Flood|The submergence of land not usually covered by water and where “stream flow is characteristically turbulent”. There are two cases of floods, the first happens when sediment is carried down a river or stream making the water appear muddy. Since sediment is present in the water, the flood becomes more powerful and damaging to nearby areas, and is usually in more rural areas. The second case happens when there is not as much sediment build up in a flood. These kinds of floods usually occur in urban settings ([Nelson](https://www.tulane.edu/~sanelson/Natural_Disasters/riversystems.htm)).||
|Swamp| “Areas of land permanently saturated, or filled, with water. Many swamps are even covered by water. There are two main types of swamps: freshwater swamps and saltwater swamps". In freshwater swamps in the United States, “cypress and tupelo trees grow and Spanish mossmay hang from the branches with tiny plants called duckweed covering the water’s surface.Shrubs and bushes may grow beneath the trees, and cypress knees may poke out as muchas 4 meters above the water” (Rutledge, et al, 2011). This environment differs from that of saltwater swamps, which “form on tropical coastlines beginning with bare flats of mud and sand that are thinly covered by seawater during high tides, and plants that are able to tolerate tidal flooding, such as mangrove trees, begin to grow and soon form thickets of roots and branches” ([National Geographic Society](https://www.nationalgeographic.org/encyclopedia/swamp/)).|
|River|Rivers tend to be long fast flowing of water from a high to a low elevation. When classifying a river, there should be a noticeable flow and is not standstill like a lake or a pond. The direction of the flow will tend to be from left to right or vice versa, because that is how rivers flow since waterbodies like sea or lake flow towards the coast. The colors of rivers can change due to the weather and soil makeup but blue, brown, and grey are the main ones. ([National Geographic Society](https://www.nationalgeographic.org/encyclopedia/river/))|
|Sea|Sea or Oceans will have rough water and a noticeable horizon to indicate the mass size ofthe waterbody. The direction of the water will be from top to bottom of a photo (vice versa, depending on tide) which can be shown through rough waves. Seas consists of white water on the coastline due to waves crashing in and can be used as a noticeable sight of a sea. The white water also has a foam texture to it due to the bubbles formed. Seas will always be rough and will have a sharp look to it from the many directions and waves hitting each other.|
|Snow|A white form of precipitation covering the top of most things in the area. Snow will always be white and has a fluffy look to it. Often seen in piles and lumps, it is never chunky but smooth and fragile. The snow will cover the whole surface and will be found on top of other objects. The snow also carries no reflection. No other water body will look like snow since white water has bubbles and snow is completely white. ([National Geographic Society](https://www.nationalgeographic.com/environment/article/avalanche-winter-general))|

<img width="500" alt="Screen Shot 2021-07-15 at 2 40 11 PM" src="https://user-images.githubusercontent.com/87332442/125840255-f9fa0e68-891c-4060-a935-ca3bb8f41ab2.png">

<p float="left">
  <img src="https://user-images.githubusercontent.com/87332442/125840255-f9fa0e68-891c-4060-a935-ca3bb8f41ab2.png" width="33%" />
  <img src="https://user-images.githubusercontent.com/87332442/126213674-b3fc8734-f075-4c37-8f30-796901351e22.png" width="33%" /> 
  <img src="https://user-images.githubusercontent.com/87332442/126213674-b3fc8734-f075-4c37-8f30-796901351e22.png" width="33%" />
</p>




## ATeX Related Projects
* [ATLANTIS](https://github.com/smhassanerfani/atlantis) is a code used for downloading images from [Flickr](https://www.flickr.com) 

### Citations
Mohammad

