"""Christmas/winter language-model tools with a Google Colab entry point.

Open Christmasmodel_Colab.ipynb for the current workflow. It fetches the source
from GitHub, pins a code revision in Drive, and offers small scratch, matched
scratch, and pretrained branches. Adaptation text includes works through 1920.
The pretrained branch carries outside knowledge; the small scratch branch learns
both tokenizer and model weights solely from the prepared historical corpus.

The stories subcommand delegates to the story_* modules in this repository.
Its notebook default displays a plan, without starting training. Legacy commands
below retain their original sources and checkpoint formats for existing runs.
Installation, tests, downloads, training, and export belong in Colab.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# Current workflow: use Christmasmodel_Colab.ipynb with the repository modules.
# This default only displays a plan; it does not launch the legacy objectives.
NOTEBOOK_ARGS = ["stories", "plan", "--config", "configs/story-scratch.json"]
NOTEBOOK_REQUIREMENTS = [
    "torch>=2.6,<3", "transformers==4.57.6", "tokenizers==0.22.2", "accelerate==1.12.0",
    "huggingface-hub==0.36.2", "numpy>=1.26,<3", "requests>=2.32,<3", "safetensors>=0.4",
]
# These fingerprints preserve compatibility with already-prepared corpus.py runs.
ORIGINAL_SOURCES_SHA256 = "eb8ba42bffccb2073023e77355c9e2dccd6d99aa79e8a36dc11c41eb6485abe1"
ORIGINAL_CORPUS_SHA256 = "47568b74f9fefff60759464ccb36d610dd5b02b2772f2de40be6f9f736ef2dfa"

# Complete source list, also maintained in corpus.py for repository users.
sources = [
  "https://www.gutenberg.org/cache/epub/46/pg46.txt",       #A Christmas Carol in Prose; Being a Ghost Story of Christmas by Charles Dickens
  "https://www.gutenberg.org/cache/epub/28198/pg28198.txt", #A Budget of Christmas Tales by Charles Dickens and Others by Charles Dickens et al.
  "https://www.gutenberg.org/cache/epub/653/pg653.txt",     #The Chimes by Charles Dickens
  "https://www.gutenberg.org/cache/epub/644/pg644.txt",     #The Haunted Man and the Ghost's Bargain by Charles Dickens
  "https://www.gutenberg.org/cache/epub/721/pg721.txt",     #The Birds' Christmas Carol by Kate Douglas Smith Wiggin
  "https://www.gutenberg.org/cache/epub/10679/pg10679.txt", #The Story of the Other Wise Man by Henry Van Dyke
  "https://www.gutenberg.org/cache/epub/8694/pg8694.txt",   #The Abbot's Ghost, or Maurice Treherne's Temptation: A Christmas Story by Alcott
  "https://www.gutenberg.org/cache/epub/31103/pg31103.txt", #A Christmas Greeting: A Series of Stories by H. C. Andersen
  "https://www.gutenberg.org/cache/epub/25927/pg25927.txt", #The Christmas Story from David Harum by Edward Noyes Westcott
  "https://www.gutenberg.org/cache/epub/1415/pg1415.txt",   #Doctor Marigold by Charles Dickens
  "https://www.gutenberg.org/cache/epub/16052/pg16052.txt", #The Brownies and Other Tales by Juliana Horatia Ewing
  "https://www.gutenberg.org/cache/epub/1394/pg1394.txt",   #The Holly-Tree by Charles Dickens
  "https://www.gutenberg.org/cache/epub/47333/pg47333.txt", #Christmas Stories from French and Spanish Writers by Antoinette Ogden
  "https://www.gutenberg.org/cache/epub/58409/pg58409.txt", #Cossack Tales by Nikolai Vasilevich Gogol
  "https://www.gutenberg.org/cache/epub/28098/pg28098.txt", #Holiday Tales: Christmas in the Adirondacks by W. H. H. Murray
  "https://www.gutenberg.org/cache/epub/16134/pg16134.txt", #The First Christmas Tree: A Story of the Forest by Henry Van Dyke
  "https://www.gutenberg.org/cache/epub/20018/pg20018.txt", #Evenings at Donaldson Manor; Or, The Christmas Guest by Maria J. McIntosh
  "https://www.gutenberg.org/cache/epub/58558/pg58558.txt", #Christmas at Thompson Hall by Anthony Trollope
  "https://www.gutenberg.org/cache/epub/20251/pg20251.txt", #Christmas Comes but Once a Year by John Leighton
  "https://www.gutenberg.org/cache/epub/10723/pg10723.txt", #Betty's Bright Idea; Deacon Pitkin's Farm; and the First Christmas of New…
  "https://www.gutenberg.org/cache/epub/19608/pg19608.txt", #The Story of the Other Wise Man by Henry Van Dyke
  "https://www.gutenberg.org/cache/epub/42961/pg42961.txt", #The House With Sixty Closets: A Christmas Story for Young Folks and Old…
  "https://www.gutenberg.org/cache/epub/61193/pg61193.txt", #Mr. Pickwick's Christmas by Charles Dickens
  "https://www.gutenberg.org/cache/epub/31993/pg31993.txt", #'Round the yule-log: Christmas in Norway by Peter Christen Asbjørnsen
  "https://www.gutenberg.org/cache/epub/14624/pg14624.txt", #Santa Claus's Partner by Thomas Nelson Page
  "https://www.gutenberg.org/cache/epub/38279/pg38279.txt", #How John Norton the Trapper Kept His Christmas by W. H. H. Murray
  "https://www.gutenberg.org/cache/epub/43190/pg43190.txt", #Blade-O'-Grass. Golden Grain. and Bread and Cheese and Kisses. by B. L. Farjeon
  "https://www.gutenberg.org/cache/epub/26779/pg26779.txt", #The Ghost by William Douglas O'Connor
  "https://www.gutenberg.org/cache/epub/64152/pg64152.txt", #The Strange Visitation by Marie Corelli
  "https://www.gutenberg.org/cache/epub/44547/pg44547.txt", #Polly: A Christmas Recollection by Thomas Nelson Page
  "https://www.gutenberg.org/cache/epub/41831/pg41831.txt", #Betty Leicester's Christmas by Sarah Orne Jewett
  "https://www.gutenberg.org/cache/epub/48747/pg48747.txt", #Who Ate the Pink Sweetmeat? And Other Christmas Stories by Susan Coolidge et al.
  "https://www.gutenberg.org/cache/epub/28307/pg28307.txt", #A Christmas Accident and Other Stories by Annie Eliot Trumbull
  "https://www.gutenberg.org/cache/epub/32520/pg32520.txt", #Hildegarde's Harvest by Laura Elizabeth Howe Richards
  "https://www.gutenberg.org/cache/epub/34665/pg34665.txt", #Christmas Penny Readings: Original Sketches for the Season by George Manville Fenn
  "https://www.gutenberg.org/cache/epub/14534/pg14534.txt", #Christmas with Grandma Elsie by Martha Finley
  "https://www.gutenberg.org/cache/epub/70198/pg70198.txt", #Seven Xmas Eves : Being the romance of a social evolution by Richard Dehan et al.
  "https://www.gutenberg.org/cache/epub/23063/pg23063.txt", #The Lost Child by François Coppée
  "https://www.gutenberg.org/cache/epub/4384/pg4384.txt",   #The Lost Word: A Christmas Legend of Long Ago by Henry Van Dyke
  "https://www.gutenberg.org/cache/epub/56410/pg56410.txt", #Ruby: A Story of the Australian Bush by Molly E. Jamieson
  "https://www.gutenberg.org/cache/epub/74849/pg74849.txt", #The Christmas earnings : Or, Ethel Fletcher's temptation by Lucy Ellen Guernsey
  "https://www.gutenberg.org/cache/epub/3719/pg3719.txt",   #The Mistletoe Bough by Anthony Trollope
  "https://www.gutenberg.org/cache/epub/72586/pg72586.txt", #Ginevra : or, The old oak chest, a Christmas story by Susan E. Wallace
  "https://www.gutenberg.org/cache/epub/20200/pg20200.txt", #Christmas, A Happy Time by Alicia Catherine Mant
  "https://www.gutenberg.org/cache/epub/20180/pg20180.txt", #Mr. Kris Kringle: A Christmas Tale by S. Weir Mitchell
  "https://www.gutenberg.org/cache/epub/56599/pg56599.txt", #The Fortunes of the Colville Family; or, A Cloud with its Silver Lining by Smedley
  "https://www.gutenberg.org/cache/epub/55281/pg55281.txt", #The Christmas Dream of Little Charles by Redfield, Justus Starr
  "https://www.gutenberg.org/cache/epub/77220/pg77220.txt", #Tabby's travels : or the holiday adventures of a kitten by Lucy Ellen Guernsey
  "https://www.gutenberg.org/cache/epub/53920/pg53920.txt", #Kittyboy's Christmas by Amy Ella Blanchard
  "https://www.gutenberg.org/cache/epub/64117/pg64117.txt", #The Christmas Holly by Marion Harland
  "https://www.gutenberg.org/cache/epub/75031/pg75031.txt", #Corrie by active approximately 1876-1880 Ruth Lynn
  "https://www.gutenberg.org/cache/epub/22665/pg22665.txt", #Christian Gellert's Last Christmas by Berthold Auerbach
  "https://www.gutenberg.org/cache/epub/74860/pg74860.txt", #Elsie : a Christmas story by Alexander Lange Kielland
  "https://www.gutenberg.org/cache/epub/64094/pg64094.txt", #Christmas on Wheels by Willis Boyd Allen
  "https://www.gutenberg.org/cache/epub/5591/pg5591.txt",   #The Nuts: A Christmas Story for my Children and Grandchildren by Georg Ebers
  "https://www.gutenberg.org/cache/epub/35397/pg35397.txt", #Christmas Stories by Edward Berens
  "https://www.gutenberg.org/cache/epub/67194/pg67194.txt", #Christmas Holidays; or, a Visit at Home by Anonymous
  "https://www.gutenberg.org/cache/epub/71698/pg71698.txt", #Christmas at Cedar Hill : A holiday story-book by Lucy Ellen Guernsey
  "https://www.gutenberg.org/cache/epub/79517/pg79517.txt", #Christie's Christmas by Pansy
  "https://www.gutenberg.org/cache/epub/65830/pg65830.txt", #Two Christmas Stories: Sam Franklin's Savings-Bank; A Miserable Christmas and…
  "https://www.gutenberg.org/cache/epub/72610/pg72610.txt", #The Christmas gift : A story for little folks by Oliver Optic
  "https://www.gutenberg.org/cache/epub/70279/pg70279.txt", #Bertha Weisser's wish : A Christmas story by Mary L. Bissell
  "https://www.gutenberg.org/cache/epub/70291/pg70291.txt", #X-mas sketches from the Dartmouth Literary Monthly by Edwin Osgood Grover
  "https://www.gutenberg.org/cache/epub/68424/pg68424.txt", #Magic words: A tale for Christmas time by Emilie Maceroni
  "https://www.gutenberg.org/cache/epub/72621/pg72621.txt", #Our Christmas party by Edwin Hodder
  "https://www.gutenberg.org/cache/epub/1467/pg1467.txt",   #Some Christmas Stories by Charles Dickens
  "https://www.gutenberg.org/cache/epub/809/pg809.txt",     #Holiday Romance by Charles Dickens
  "https://www.gutenberg.org/cache/epub/678/pg678.txt",     #The Cricket on the Hearth: A Fairy Tale of Home by Charles Dickens
  "https://www.gutenberg.org/cache/epub/23540/pg23540.txt", #The Twin Cousins by Sophie May
  "https://www.gutenberg.org/cache/epub/52935/pg52935.txt", #Mr. Blake's Walking-Stick: A Christmas Story for Boys and Girls by Edward Eggleston
  "https://www.gutenberg.org/cache/epub/40269/pg40269.txt", #At the Black Rocks by Edward A. Rand
  "https://www.gutenberg.org/cache/epub/26015/pg26015.txt", #A Christmas Posy by Mrs. Molesworth
  "https://www.gutenberg.org/cache/epub/14202/pg14202.txt", #Little Prudy's Sister Susy by Sophie May
  "https://www.gutenberg.org/cache/epub/45666/pg45666.txt", #Little Peter: A Christmas Morality for Children of any Age by Lucas Malet
  "https://www.gutenberg.org/cache/epub/38564/pg38564.txt", #Happy Hearts by June Isle
  "https://www.gutenberg.org/cache/epub/20425/pg20425.txt", #The Peace Egg and Other tales by Juliana Horatia Ewing
  "https://www.gutenberg.org/cache/epub/48325/pg48325.txt", #Karl Krinken, His Christmas Stocking by Susan Warner and Anna Bartlett Warner
  "https://www.gutenberg.org/cache/epub/39210/pg39210.txt", #The Ice Queen by Ernest Ingersoll
  "https://www.gutenberg.org/cache/epub/36431/pg36431.txt", #The Palace in the Garden by Mrs. Molesworth
  "https://www.gutenberg.org/cache/epub/34045/pg34045.txt", #A Christmas Child: A Sketch of a Boy-Life by Mrs. Molesworth
  "https://www.gutenberg.org/cache/epub/43763/pg43763.txt", #Snap-Dragons; Old Father Christmas by Juliana Horatia Ewing
  "https://www.gutenberg.org/cache/epub/69782/pg69782.txt", #A Southern Cross fairy tale by Kate McCosh Clark
  "https://www.gutenberg.org/cache/epub/58762/pg58762.txt", #The New-Year's Bargain by Susan Coolidge
  "https://www.gutenberg.org/cache/epub/47897/pg47897.txt", #The yule log : a series of stories for the young by Georgianna M. Bishop
  "https://www.gutenberg.org/cache/epub/71907/pg71907.txt", #Arthur Glyn : and other stories by Ruth Lamb
  "https://www.gutenberg.org/cache/epub/57413/pg57413.txt", #Down the Snow Stairs; Or, From Good-Night to Good-Morning by Alice Corkran
  "https://www.gutenberg.org/cache/epub/13379/pg13379.txt", #The Two Elsies by Martha Finley
  "https://www.gutenberg.org/cache/epub/54803/pg54803.txt", #Playing Santa Claus, and Other Christmas Tales by Sarah P. Doughty
  "https://www.gutenberg.org/cache/epub/2731/pg2731.txt",   #The Christmas Books of Mr. M.A. Titmarsh by William Makepeace Thackeray
  "https://www.gutenberg.org/cache/epub/40723/pg40723.txt", #The Battle of Life: A Love Story by Charles Dickens
  "https://www.gutenberg.org/cache/epub/1392/pg1392.txt",   #The Seven Poor Travellers by Charles Dickens
  "https://www.gutenberg.org/cache/epub/79846/pg79846.txt", #The written leaves : A story of the new year and other holiday stories
  ##### Less Strict Time Period Adherence (1900-1920) #####
  "https://www.gutenberg.org/cache/epub/64103/pg64103.txt", #Little Paulina: Christmas in Russia by Mary Cowden Clarke
  "https://www.gutenberg.org/cache/epub/70097/pg70097.txt", #A new note in the Christmas Carol by Clara Wood Mingins
  "https://www.gutenberg.org/cache/epub/70521/pg70521.txt", #The hermit's Christmas by David De Forest Burrell
  "https://www.gutenberg.org/cache/epub/72453/pg72453.txt", #The Squire's young folk : A Christmas story by Eleanora H. Stooke
  "https://www.gutenberg.org/cache/epub/78374/pg78374.txt", #Little Merry Christmas by Winifred Arnold
  "https://www.gutenberg.org/cache/epub/70294/pg70294.txt", #The First Church's Christmas barrel by Caroline Abbot Stanley
  "https://www.gutenberg.org/cache/epub/67255/pg67255.txt", #The Story of Zephyr: A Christmas Story by Jeanie Oliver Davidson Smith
  "https://www.gutenberg.org/cache/epub/70273/pg70273.txt", #The boy who brought Christmas by Alice Morgan
  "https://www.gutenberg.org/cache/epub/67034/pg67034.txt", #The Child and the Dream: A Christmas Story by Marion Cook Stow
  "https://www.gutenberg.org/cache/epub/69685/pg69685.txt", #Just sweethearts: A Christmas love story by Harry Stillwell Edwards
  "https://www.gutenberg.org/cache/epub/69068/pg69068.txt", #The country Christmas by Frances Margaret Fox
  "https://www.gutenberg.org/cache/epub/12881/pg12881.txt", #Mrs. Budlong's Christmas Presents by Rupert Hughes
  "https://www.gutenberg.org/cache/epub/74537/pg74537.txt", #The wonderful Christmas in Pumpkin Delight Lane by Sarah J. Prichard
  "https://www.gutenberg.org/cache/epub/15826/pg15826.txt", #Uncle Noah's Christmas Inspiration by Leona Dalrymple
  "https://www.gutenberg.org/cache/epub/17510/pg17510.txt", #When the Yule Log Burns: A Christmas Story by Leona Dalrymple
  "https://www.gutenberg.org/cache/epub/28110/pg28110.txt", #Jimsy: The Christmas Kid by Leona Dalrymple
  "https://www.gutenberg.org/cache/epub/45103/pg45103.txt", #In the Heart of the Christmas Pines by Leona Dalrymple
  "https://www.gutenberg.org/cache/epub/69854/pg69854.txt", #The Christmas Makers' Club by Edith A. Sawyer
  "https://www.gutenberg.org/cache/epub/15936/pg15936.txt", #The Sad Shepherd: A Christmas Story by Henry Van Dyke
  "https://www.gutenberg.org/cache/epub/704/pg704.txt",     #The Mansion by Henry Van Dyke
  "https://www.gutenberg.org/cache/epub/14946/pg14946.txt", #The Blossoming Rod by Mary Stewart Cutting
  "https://www.gutenberg.org/cache/epub/71547/pg71547.txt", #A journey in search of Christmas by Owen Wister
  "https://www.gutenberg.org/cache/epub/68590/pg68590.txt", #The Christmas Bishop by Winifred Margaretta Kirkland
  "https://www.gutenberg.org/cache/epub/71029/pg71029.txt", #Jasper's old shed, and how the light shone in by A. M. Coker
  "https://www.gutenberg.org/cache/epub/23278/pg23278.txt", #Janice Day at Poketown by Helen Beecher Long
  "https://www.gutenberg.org/cache/epub/59594/pg59594.txt", #Betty's Virginia Christmas by Molly Elliot Seawell
  "https://www.gutenberg.org/cache/epub/75689/pg75689.txt", #Beth's wonder-winter : A story by Marion Ames Taggart
  "https://www.gutenberg.org/cache/epub/30907/pg30907.txt", #Rosemary in Search of a Father by C. N. Williamson and A. M. Williamson
  "https://www.gutenberg.org/cache/epub/39753/pg39753.txt", #The Misfit Christmas Puddings by Consolation Club
  "https://www.gutenberg.org/cache/epub/27697/pg27697.txt", #The Mouse and The Moonbeam by Eugene Field
  "https://www.gutenberg.org/cache/epub/19384/pg19384.txt", #On Christmas Day In The Evening by Grace S. Richmond
  "https://www.gutenberg.org/cache/epub/53402/pg53402.txt", #Brotherly House by Grace S. Richmond
  "https://www.gutenberg.org/cache/epub/20187/pg20187.txt", #On Christmas Day in the Morning by Grace S. Richmond
  "https://www.gutenberg.org/cache/epub/67105/pg67105.txt", #The Animals' Christmas Tree by John P. Peters
  "https://www.gutenberg.org/cache/epub/64107/pg64107.txt", #Christmas in Sweden; or, A festival of light by Sarah Gertrude Pomeroy
  "https://www.gutenberg.org/cache/epub/33666/pg33666.txt", #A Captured Santa Claus by Thomas Nelson Page
  "https://www.gutenberg.org/cache/epub/25896/pg25896.txt", #Tommy Trot's Visit to Santa Claus by Thomas Nelson Page
  "https://www.gutenberg.org/cache/epub/23511/pg23511.txt", #The Christmas Peace by Thomas Nelson Page
  "https://www.gutenberg.org/cache/epub/69416/pg69416.txt", #Christmas stories by Georgene Faulkner
  "https://www.gutenberg.org/cache/epub/5303/pg5303.txt",   #The Little Hunchback Zia by Frances Hodgson Burnett
  "https://www.gutenberg.org/cache/epub/479/pg479.txt",     #Little Lord Fauntleroy by Frances Hodgson Burnett
  "https://www.gutenberg.org/cache/epub/460/pg460.txt",     #The Dawn of a To-morrow by Frances Hodgson Burnett
  "https://www.gutenberg.org/cache/epub/38555/pg38555.txt", #Dorothy Dale in the City by Margaret Penrose
  "https://www.gutenberg.org/cache/epub/37152/pg37152.txt", #A Maid of Many Moods by Virna Sheard
  "https://www.gutenberg.org/cache/epub/5949/pg5949.txt",   #Beasley's Christmas Party by Booth Tarkington
  "https://www.gutenberg.org/cache/epub/14723/pg14723.txt", #How It Happened by Kate Langley Bosher
  "https://www.gutenberg.org/cache/epub/12971/pg12971.txt", #The Man in Lonely Land by Kate Langley Bosher
  "https://www.gutenberg.org/cache/epub/44796/pg44796.txt", #The Prodigal Village: A Christmas Tale by Irving Bacheller
  "https://www.gutenberg.org/cache/epub/20516/pg20516.txt", #Christmas: A Story by Zona Gale
  "https://www.gutenberg.org/cache/epub/14526/pg14526.txt", #The Little City of Hope: A Christmas Story by F. Marion Crawford
  "https://www.gutenberg.org/cache/epub/14606/pg14606.txt", #And Thus He Came: A Christmas Fantasy by Cyrus Townsend Brady
  "https://www.gutenberg.org/cache/epub/75482/pg75482.txt", #Maisie's merry Christmas by Nina Rhoades
  "https://www.gutenberg.org/cache/epub/55143/pg55143.txt", #The Story of the Mince Pie by Josephine Scribner Gates
  "https://www.gutenberg.org/cache/epub/19988/pg19988.txt", #Little Maid Marian by Amy Ella Blanchard
  "https://www.gutenberg.org/cache/epub/24310/pg24310.txt", #Candle and Crib by K. F. Purdon
  "https://www.gutenberg.org/cache/epub/64124/pg64124.txt", #Santa Claus' Sweetheart by Imogen Clark
  "https://www.gutenberg.org/cache/epub/36255/pg36255.txt", #The Corner House Girls' Odd Find by Grace Brooks Hill
  "https://www.gutenberg.org/cache/epub/16454/pg16454.txt", #The Upas Tree: A Christmas Story for all the Year by Florence L. Barclay
  "https://www.gutenberg.org/cache/epub/31997/pg31997.txt", #Miss Muffet's Christmas Party by Samuel McChord Crothers
  "https://www.gutenberg.org/cache/epub/5963/pg5963.txt",   #Mr. Bingle by George Barr McCutcheon
  "https://www.gutenberg.org/cache/epub/20213/pg20213.txt", #Peace on Earth, Good-will to Dogs by Eleanor Hallowell Abbott
  "https://www.gutenberg.org/cache/epub/69097/pg69097.txt", #Christmas in Austria; or, Fritzl's friends by Frances Bartlett
  "https://www.gutenberg.org/cache/epub/69601/pg69601.txt", #The escape of Alice: A Christmas fantasy by Vincent Starrett
  "https://www.gutenberg.org/cache/epub/35178/pg35178.txt", #Ben Pepper by Margaret Sidney
  "https://www.gutenberg.org/cache/epub/14573/pg14573.txt", #The Truce of God by Mary Roberts Rinehart
  "https://www.gutenberg.org/cache/epub/34465/pg34465.txt", #A Little Book of Christmas by John Kendrick Bangs
  "https://www.gutenberg.org/cache/epub/71759/pg71759.txt", #Granfer, and One Christmas time by Eleanora H. Stooke
  "https://www.gutenberg.org/cache/epub/42919/pg42919.txt", #Angel Unawares: A Story of Christmas Eve by C. N. Williamson and A. M. Williamson
  "https://www.gutenberg.org/cache/epub/1902/pg1902.txt",   #The Old Peabody Pew: A Christmas Romance of a Country Church by Wiggin
  "https://www.gutenberg.org/cache/epub/20453/pg20453.txt", #The Christmas Child by Hesba Stretton
  "https://www.gutenberg.org/cache/epub/17743/pg17743.txt", #Rosemary: A Christmas story by C. N. Williamson and A. M. Williamson
  "https://www.gutenberg.org/cache/epub/40401/pg40401.txt", #Three Little Women's Success: A Story for Girls by Gabrielle E. Jackson
  "https://www.gutenberg.org/cache/epub/17456/pg17456.txt", #The Romance of a Christmas Card by Kate Douglas Smith Wiggin
  "https://www.gutenberg.org/cache/epub/51942/pg51942.txt", #The Three Godfathers by Peter B. Kyne
  "https://www.gutenberg.org/cache/epub/15709/pg15709.txt", #The Christmas Angel by Abbie Farwell Brown
  "https://www.gutenberg.org/cache/epub/15044/pg15044.txt", #A Reversible Santa Claus by Meredith Nicholson
  "https://www.gutenberg.org/cache/epub/17937/pg17937.txt", #The Thin Santa Claus: The Chicken Yard That Was a Christmas Stocking by Butler
  "https://www.gutenberg.org/cache/epub/11084/pg11084.txt", #Sonny, a Christmas Guest by Ruth McEnery Stuart
  "https://www.gutenberg.org/cache/epub/50274/pg50274.txt", #The Little House by Coningsby Dawson
  "https://www.gutenberg.org/cache/epub/27612/pg27612.txt", #Christmas Eve at Swamp's End by Norman Duncan
  "https://www.gutenberg.org/cache/epub/25869/pg25869.txt", #Patty's Success by Carolyn Wells
  "https://www.gutenberg.org/cache/epub/42923/pg42923.txt", #The Doctor's Christmas Eve by James Lane Allen
  "https://www.gutenberg.org/cache/epub/9179/pg9179.txt",   #The Bride of the Mistletoe by James Lane Allen
  "https://www.gutenberg.org/cache/epub/10707/pg10707.txt", #A Christmas Mystery: The Story of Three Wise Men by William John Locke
  "https://www.gutenberg.org/cache/epub/41604/pg41604.txt", #Miss Santa Claus of the Pullman by Annie F. Johnston
  "https://www.gutenberg.org/cache/epub/40463/pg40463.txt", #The Little Colonel's Holidays by Annie F. Johnston
  "https://www.gutenberg.org/cache/epub/26215/pg26215.txt", #The Little Colonel's Christmas Vacation by Annie F. Johnston
  "https://www.gutenberg.org/cache/epub/17770/pg17770.txt", #Christmas Stories And Legends by Phebe A. Curtiss
  "https://www.gutenberg.org/cache/epub/14835/pg14835.txt", #The Burglar and the Blizzard: A Christmas Story by Alice Duer Miller
  "https://www.gutenberg.org/cache/epub/36089/pg36089.txt", #Back at School with the Tucker Twins by Nell Speed
  "https://www.gutenberg.org/cache/epub/34571/pg34571.txt", #The Pearl Story Book: Stories and Legends of Winter, Christmas, and New Year's…
  "https://www.gutenberg.org/cache/epub/22696/pg22696.txt", #Colonel Crockett's Co-operative Christmas by Rupert Hughes
  "https://www.gutenberg.org/cache/epub/23569/pg23569.txt", #Christmas Holidays at Merryvale by Alice Hale Burnett
  "https://www.gutenberg.org/cache/epub/28308/pg28308.txt", #The Children's Book of Christmas Stories by Asa Don Dickinson and Ada M. Skinner
  "https://www.gutenberg.org/cache/epub/519/pg519.txt",     #A Kidnapped Santa Claus by L. Frank Baum
  "https://www.gutenberg.org/cache/epub/27741/pg27741.txt", #Colonel Carter's Christmas and The Romance of an Old-Fashioned Gentleman by Smith
  "https://www.gutenberg.org/cache/epub/520/pg520.txt",     #The Life and Adventures of Santa Claus by L. Frank Baum
]

EOT, PAD = "<|endoftext|>", "<|pad|>"
PROMPTS = [
    "On Christmas Eve, the snow fell softly over the village. At the window,",
    '\"Come in, my dear,\" said the old woman, \"and warm yourself by the fire.',
    "The little girl had saved one candle for her father. When the clock struck",
]
CLEANING_VERSION = 1



# Reviewed pre-1920 additions, used ONLY by finetune-winter. All URLs,
# bibliographic evidence, exact boundaries and content hashes are inline for Colab.
WINTER_SOURCES = [{'id': '32090',
  'title': 'The Curlytops Snowed In; Or, Grand Fun with Skates and Sleds',
  'url': 'https://www.gutenberg.org/cache/epub/32090/pg32090.txt',
  'date': '1918',
  'date_evidence': 'Original copyright 1918 in Gutenberg front matter; transcribed printing also shows 1941. '
                   'Original year qualifies; exclude later publisher lists.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/32090/pg32090-images.html'],
  'note': 'Baby-talk, animal antics, rescue subplot, sequel references and publisher advertisements. Sampled '
          'passages only; not a fully reviewed curriculum.',
  'selections': [{'start': '^CHAPTER I$',
                  'end': '^THE END\\.?$',
                  'natural_end': True,
                  'sha256': 'cbd344965921f93ae939981d69a7a0fe4cbfc6b2dad391c496c6f1e8bb28d665',
                  'words': 45084}]},
 {'id': '17761',
  'title': "Six little Bunkers at Grandpa Ford's",
  'url': 'https://www.gutenberg.org/cache/epub/17761/pg17761.txt',
  'date': '1918',
  'date_evidence': 'Copyright 1918 in Gutenberg front matter.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/17761/pg17761-images.html'],
  'note': 'Six children plus adults increase cast size; repeated series recap, mystery thread, dialect, '
          'publisher advertisements. Select bounded exchanges.',
  'selections': [{'start': '^CHAPTER I$',
                  'end': '^THE END\\.?$',
                  'natural_end': True,
                  'sha256': '3a0d4ce44a548cc5e7c4a6c436fd0e7e39b4f1311a2a0d9ec2e8fc335c129ee3',
                  'words': 42470}]},
 {'id': '6055',
  'title': 'The Bobbsey Twins at Snow Lodge',
  'url': 'https://www.gutenberg.org/cache/epub/6055/pg6055.txt',
  'date': '1913',
  'date_evidence': 'Copyright 1913 in Gutenberg front matter.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/6055/pg6055-images.html'],
  'note': 'Mystery plot, period dialect and stereotypes, series recap; review scene boundaries rather than '
          'blindly taking windows.',
  'selections': [{'start': '^CHAPTER I$',
                  'end': '^THE END\\.?$',
                  'natural_end': True,
                  'sha256': 'b021939b59798af70a9d282cd8c48887b5bc5c96174fdb014458775abdbf2038',
                  'words': 36332}]},
 {'id': '2786',
  'title': 'Jack and Jill',
  'url': 'https://www.gutenberg.org/cache/epub/2786/pg2786.txt',
  'date': '1880',
  'date_evidence': 'Dedication dated 1880; Gutenberg catalog describes 1879-1880 serialization.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/2786/pg2786-images.html',
                   'https://www.gutenberg.org/ebooks/2786'],
  'note': 'Longer sentences and larger cast than youngest-reader books; later chapters move into '
          'spring/summer. Prefer winter chapters and complete scenes.',
  'selections': [{'start': '^Chapter I\\. The Catastrophe$',
                  'end': '^Chapter XIII\\. Jack Has a Mystery$',
                  'natural_end': False,
                  'sha256': 'f879a279321fd18a0eedccec805fcfe03ddcd9cf00d5155f70a2cb09c76e30bb',
                  'words': 43162}]},
 {'id': '5347',
  'title': 'Understood Betsy',
  'url': 'https://www.gutenberg.org/cache/epub/5347/pg5347.txt',
  'date': 'by 1917',
  'date_evidence': 'NYPL scanned edition cataloged c1917 and visible copyright evidence says 1917; Gutenberg '
                   'automated summary says 1916. Pre-1920 status verified; exact earliest issue date not '
                   'resolved.',
  'date_sources': ['https://archive.org/details/understoodbetsy00fish',
                   'https://www.gutenberg.org/ebooks/5347'],
  'note': 'Not winter throughout; intrusive narrator and longer introspective passages. Select concrete '
          'household scenes.',
  'selections': [{'start': '^CHAPTER I$',
                  'end': '^CHAPTER VIII$',
                  'natural_end': False,
                  'sha256': 'f19e1301e964e19bbf0398ea42a535973a412c9c5386be7881f2dbf39190e461',
                  'words': 27805},
                 {'start': '^CHAPTER XI$',
                  'end': '^THE END\\.$',
                  'natural_end': True,
                  'sha256': 'd2edb98b7e0acd489229f9282efc32e20e2a11a1bda87450c691b040142b7dd7',
                  'words': 6493}]},
 {'id': '514',
  'title': 'Little Women',
  'url': 'https://www.gutenberg.org/cache/epub/514/pg514.txt',
  'date': '1868-1869',
  'date_evidence': 'Library of Congress dates first volume to 1868 and second to the following year.',
  'date_sources': ['https://www.loc.gov/item/today-in-history/november-29/'],
  'note': 'Much of the novel is not winter; varied registers, verse, embedded theatrical/newspaper material '
          'and adult courtship. Use selected winter/domestic chapters rather than bulk inclusion.',
  'selections': [{'start': '^CHAPTER ONE$',
                  'end': '^CHAPTER TEN$',
                  'natural_end': False,
                  'sha256': 'e2fd5f41479e714b8771123907868c38c25d2af69c4dd715d1d071dfcbb7bdb5',
                  'words': 37109},
                 {'start': '^CHAPTER FIFTEEN$',
                  'end': '^PART 2$',
                  'natural_end': False,
                  'sha256': '39657afbc52999e6d1483cd8bad6cdaeda34b3b16b4e978f9e82af23160d6e52',
                  'words': 29311}]},
 {'id': '17878',
  'title': 'Bunny Brown and his Sister Sue Giving a Show',
  'url': 'https://www.gutenberg.org/cache/epub/17878/pg17878.txt',
  'date': '1919',
  'date_evidence': 'Copyright 1919 in the transcribed edition.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/17878/pg17878.txt'],
  'note': 'Series recap, theatrical mystery subplot, occasional dialect; select complete scenes rather than '
          'arbitrary windows.',
  'selections': [{'start': '^CHAPTER I$',
                  'end': '^THE END\\.$',
                  'natural_end': True,
                  'sha256': '1d6ad51e9da28bd80aec795b5919dad583c9b7822ea5c8a36e20edc8d4f6845a',
                  'words': 44349}]},
 {'id': '18420',
  'title': 'The Bobbsey Twins at Home',
  'url': 'https://www.gutenberg.org/cache/epub/18420/pg18420.txt',
  'date': '1916',
  'date_evidence': 'Copyright 1916 in the transcribed edition.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/18420/pg18420.txt'],
  'note': 'Begins before winter; multiple children, series recap, dialect and a missing-family subplot.',
  'selections': [{'start': '^CHAPTER I$',
                  'end': "^Transcriber's notes:$",
                  'natural_end': True,
                  'sha256': 'b3259b88f13941abd057e7d0d811bd6c938be9d8c0de7519c06141839bdfc215',
                  'words': 43014}]},
 {'id': '48951',
  'title': 'Dotty Dimple at School',
  'url': 'https://www.gutenberg.org/cache/epub/48951/pg48951.txt',
  'date': '1869',
  'date_evidence': 'Title page and copyright both 1869.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/48951/pg48951.txt'],
  'note': 'Child speech, moralizing and frightening peril. Same author/series family as existing sources: '
          'preserve held-out boundaries.',
  'selections': [{'start': '^CHAPTER I\\.$',
                  'end': '^Transcriber’s Notes:$',
                  'natural_end': True,
                  'sha256': '538fa197ba0230d6d34214999128c4e3acd8378e85580d9f0e596bef5d0e1763',
                  'words': 20980}]},
 {'id': '4012',
  'title': 'The Dutch Twins',
  'url': 'https://www.gutenberg.org/cache/epub/4012/pg4012.txt',
  'date': '1911',
  'date_evidence': 'NYPL scanned edition catalogued c1911; copyright inspection explicitly confirms 1911.',
  'date_sources': ['https://archive.org/details/dutchtwins00perk'],
  'note': 'Earlier chapters are not winter-specific. Educational framing and embedded folklore; teacher '
          'instructions excluded.',
  'selections': [{'start': '^VI\\n\\nTHE DAY THEY GOT THEIR SKATES$',
                  'end': '^SUGGESTIONS TO TEACHERS$',
                  'natural_end': True,
                  'sha256': '0370c4723d9b6032d4b77088b1a3a36df628f155b67b932d9d52518f4d31fe2b',
                  'words': 6801}]},
 {'id': '43638',
  'title': 'Our Little Swedish Cousin',
  'url': 'https://www.gutenberg.org/cache/epub/43638/pg43638.txt',
  'date': '1906',
  'date_evidence': 'Copyright 1906 and first impression July 1906.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/43638/pg43638.txt'],
  'note': 'Fictional educational travelogue: exclude geographical/historical exposition and unrelated '
          'seasons.',
  'selections': [{'start': '^CHAPTER I\\.$',
                  'end': '^CHAPTER IV$',
                  'natural_end': False,
                  'sha256': '51d40c4aab147a740fc5008e7f2302164913faca846d1bd491180ce91c07ff56',
                  'words': 6461}]},
 {'id': '43249',
  'title': 'Our Little Canadian Cousin',
  'url': 'https://www.gutenberg.org/cache/epub/43249/pg43249.txt',
  'date': '1904',
  'date_evidence': 'Copyright 1904, published July 1904; transcribed fifth impression June 1908.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/43249/pg43249.txt'],
  'note': 'Fictional educational travelogue; keep domestic episodes rather than all descriptive material.',
  'selections': [{'start': '^CHAPTER VI\\.$',
                  'end': '^CHAPTER X\\.$',
                  'natural_end': False,
                  'sha256': 'af1e975c6ed77cb75a5366ba4a4a04dccf4bb6341cd75c4c90d2fb81dbd03fa9',
                  'words': 7719}]},
 {'id': '42079',
  'title': 'Mari, Our Little Norwegian Cousin',
  'url': 'https://www.gutenberg.org/cache/epub/42079/pg42079.txt',
  'date': '1903',
  'date_evidence': 'Copyright 1903; later impressions dated 1908 and 1910.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/42079/pg42079.txt'],
  'note': 'Fictional educational travelogue; substantial fishing, geography and folklore digressions.',
  'selections': [{'start': '^CHAPTER VII\\.$',
                  'end': '^CHAPTER IX\\.$',
                  'natural_end': False,
                  'sha256': 'cd5503a3bb5a8e9bf60f866d5dfa8206c3170d15d8a7b4b5241bedffc633ff4b',
                  'words': 4481},
                 {'start': '^CHAPTER X\\.$',
                  'end': '^THE END\\.$',
                  'natural_end': True,
                  'sha256': 'e0d6f69f99d5dfcdc032d43b5f0f96388e4c6f545e5494d1e16348c3a62fedd4',
                  'words': 832}]},
 {'id': '43426',
  'title': 'Our Little Finnish Cousin',
  'url': 'https://www.gutenberg.org/cache/epub/43426/pg43426.txt',
  'date': '1918',
  'date_evidence': 'Copyright and first impression 1918; second 1919; transcribed third impression 1930.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/43426/pg43426.txt'],
  'note': 'Later printing: verify selected passages against an early edition before training if exact '
          'pre-1920 wording is required. Exclude political preface, exposition and later advertisements.',
  'selections': [{'start': '^CHAPTER I$',
                  'end': '^CHAPTER IV$',
                  'natural_end': False,
                  'sha256': '00a310c7fd6e53c6c0c367af2a383f51877a019c29b1cc8fee25565e28d4993f',
                  'words': 5699},
                 {'start': '^CHAPTER V$',
                  'end': '^CHAPTER VIII$',
                  'natural_end': False,
                  'sha256': '9caed20dad80097e0934d50666800699f5c240e319439fb8b5423b0f155d61cc',
                  'words': 5977}]},
 {'id': '20781',
  'title': 'Heidi',
  'url': 'https://www.gutenberg.org/cache/epub/20781/pg20781.txt',
  'date': '1915 English translation; 1919 edition',
  'date_evidence': 'Elisabeth P. Stork translation copyright 1915; Gift Edition title page and additional '
                   'illustrations 1919.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/20781/pg20781.txt'],
  'note': 'All-season novel with larger plot, religious passages and verse; select winter scenes. This date '
          'verifies the English text, not just the German original.',
  'selections': [{'start': "^IV\\n\\nIN THE GRANDMOTHER'S HUT$",
                  'end': '^V\\n\\nTWO VISITORS$',
                  'natural_end': False,
                  'sha256': '142c7b8dde31279ea419a2dd206019ab9e0c8ccc94210769aa793564ac221de1',
                  'words': 2697},
                 {'start': '^XVIII\\n\\nWINTER IN THE VILLAGE$',
                  'end': '^XX\\n\\nNEWS FROM DISTANT FRIENDS$',
                  'natural_end': False,
                  'sha256': '9b9edcfa716f8836c4c9f704ad7faddca16ecaf607a784117cd98e1d1c8c71cb',
                  'words': 3662}]},
 {'id': '5141',
  'title': 'What Katy Did at School',
  'url': 'https://www.gutenberg.org/cache/epub/5141/pg5141.txt',
  'date': '1873',
  'date_evidence': 'Tufts rare-book exhibit dates the work 1873 and its exhibited edition 1876.',
  'date_sources': ['https://exhibits.tufts.edu/spotlight/girls-books/catalog/32-2486'],
  'note': 'Boarding-school rather than home setting; letters, verse and longer sentences need selection.',
  'selections': [{'start': '^CHAPTER XI\\. CHRISTMAS BOXES\\.$',
                  'end': '^CHAPTER XIII\\. PARADISE REGAINED\\.$',
                  'natural_end': False,
                  'sha256': '789809adf17d13aa537300863f5b80cbad52b50c41ac2777d787f4565eee94ec',
                  'words': 7014}]},
 {'id': '5632',
  'title': 'Five Little Peppers Midway',
  'url': 'https://www.gutenberg.org/cache/epub/5632/pg5632.txt',
  'date': '1890',
  'date_evidence': 'New West Press identifies first publication as 1890 and its source edition as 1893; '
                   'corroborated by series bibliography.',
  'date_sources': ['https://newwestpress.com/products/copy-of-five-little-peppers-midway',
                   'https://19thcenturyjuvenileseries.com/series/flp_lothrop.html'],
  'note': 'Not winter throughout; larger cast and period dialect. Ben Pepper is already in the original '
          'training corpus.',
  'selections': [{'start': "^I\\n\\nPHRONSIE'S PIE$",
                  'end': '^XI\\n\\nPOOR POLLY!$',
                  'natural_end': False,
                  'sha256': '0054773b7e07d481c06536ee72e6efc82ce1ee5d359bd6d012292ea429d6d595',
                  'words': 27777}]},
 {'id': '15169',
  'title': 'The Bobbsey Twins in a Great City',
  'url': 'https://www.gutenberg.org/cache/epub/15169/pg15169.txt',
  'date': '1917',
  'date_evidence': 'Copyright 1917 in the transcribed edition.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/15169/pg15169.txt'],
  'note': 'More adventure and urban sightseeing than domestic fiction; lower priority than Bobbsey Twins at '
          'Home.',
  'selections': [{'start': '^CHAPTER I$',
                  'end': '^CHAPTER VI$',
                  'natural_end': False,
                  'sha256': 'ab5697d655b012568672930ddfa2ebc4f859b54d9cc2b6c4c50c7fea5d9c2a94',
                  'words': 9969},
                 {'start': '^CHAPTER XXI$',
                  'end': '^THE END$',
                  'natural_end': True,
                  'sha256': 'c8576a297e530d08d3c8ae9b94eec18d7d320e8a744a0a181e0329ff0c111ade',
                  'words': 3461}]},
 {'id': '26041',
  'title': "Aunt Jo's scrap bag, Volume 1 My boys, etc.",
  'url': 'https://www.gutenberg.org/cache/epub/26041/pg26041.txt',
  'date': '1892 edition; preface 1871-72',
  'date_evidence': 'Transcribed edition title page 1892, with preface dated Christmas Holidays 1871-72; '
                   "safely pre-1920 without inferring each story's original publication year.",
  'date_sources': ['https://www.gutenberg.org/cache/epub/26041/pg26041.txt'],
  'note': 'Use only the named complete stories. Poverty and moral instruction; retain original prose without '
          'rewriting it.',
  'selections': [{'start': "^_TESSA'S SURPRISES\\._$",
                  'end': '^_BUZZ\\._$',
                  'natural_end': True,
                  'sha256': '9e97778f9a90f05c9b4d777b3feef0710d8f3d7d9739b073100d12ac0907559e',
                  'words': 4403},
                 {'start': "^_TILLY'S CHRISTMAS\\._$",
                  'end': '^_MY LITTLE GENTLEMAN\\._$',
                  'natural_end': True,
                  'sha256': 'e7f0c730bda2adc03c5cb75311883c152547400bb672ab89afdbb4cf9e00010d',
                  'words': 1981}]},
 {'id': '57309',
  'title': "Aunt Jo's scrap-bag, Volume 3 Cupid and Chow-chow, etc.",
  'url': 'https://www.gutenberg.org/cache/epub/57309/pg57309.txt',
  'date': '1873 collection copyright',
  'date_evidence': 'Copyright registration 1873; transcribed title page 1896.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/57309/pg57309.txt'],
  'note': 'Family Christmas story with several relatives and a longer setup; do not add the unrelated '
          'stories in the volume.',
  'selections': [{'start': "^KATE'S CHOICE\\.$",
                  'end': '^THE MOSS PEOPLE\\.$',
                  'natural_end': True,
                  'sha256': '8480f05b5fba14204062fa60380131085f4fe5a27c449a7ce855d7cd58569a87',
                  'words': 5718}]},
 {'id': '27567',
  'title': "Aunt Jo's scrap-bag, Volume 6 An old-fashioned Thanksgiving, etc.",
  'url': 'https://www.gutenberg.org/cache/epub/27567/pg27567.txt',
  'date': '1882',
  'date_evidence': 'Title page and copyright 1882.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/27567/pg27567.txt'],
  'note': 'The second is Thanksgiving, not Christmas; both fit the expanded cold-season domestic scope. Most '
          'other collection contents are unrelated.',
  'selections': [{'start': '^AN OLD-FASHIONED THANKSGIVING\\.$',
                  'end': '^HOW IT ALL HAPPENED\\.$',
                  'natural_end': True,
                  'sha256': 'a55b444f2e0a80cfc572dd2a3ffdef9c03b6ca0cc6f6e4e9cfac9fdd9212ec72',
                  'words': 7279},
                 {'start': '^HOW IT ALL HAPPENED\\.$',
                  'end': "^THE DOLLS' JOURNEY FROM MINNESOTA TO MAINE\\.$",
                  'natural_end': True,
                  'sha256': '390dafbccba5930b78d36827a4fa4b0b6b51048ea01b29ec1ca06d8f5c8d63a7',
                  'words': 3814}]},
 {'id': '10360',
  'title': "Kitty's Class Day and Other Stories",
  'url': 'https://www.gutenberg.org/cache/epub/10360/pg10360.txt',
  'date': '1882 collection',
  'date_evidence': 'Front matter identifies original publication as Proverb Stories, 1882.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/10360/pg10360.txt'],
  'note': 'Adult visitors, romance and longer dialogue; more demanding than the youngest-reader family '
          'books.',
  'selections': [{'start': '^A COUNTRY CHRISTMAS$',
                  'end': '^ON PICKET DUTY$',
                  'natural_end': True,
                  'sha256': '684564720bbcf0fd0a6edd774f31a6c5d986ee65309894fcba46175b4ca18a5a',
                  'words': 11177}]},
 {'id': '40682',
  'title': "Lulu's Library, Volume 1 (of 3)",
  'url': 'https://www.gutenberg.org/cache/epub/40682/pg40682.txt',
  'date': '1885 copyright; 1886 edition',
  'date_evidence': 'Copyright and preface 1885; title page 1886.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/40682/pg40682.txt'],
  'note': 'Contains a dream/fantasy sequence inside a family Christmas story; keep this distinction when '
          'reviewing coherence examples.',
  'selections': [{'start': '^ A CHRISTMAS DREAM, AND HOW IT CAME TRUE\\.$',
                  'end': '^ THE CANDY COUNTRY\\.$',
                  'natural_end': True,
                  'sha256': 'ba5a5a235c0a4015b79ca2aee8b8ae30a4e828497e94559453de82be0192a241',
                  'words': 5504}]},
 {'id': '22234',
  'title': "Aunt Jo's scrap-bag, Volume 5 Jimmy's cruise in the Pinafore, etc.",
  'url': 'https://www.gutenberg.org/cache/epub/22234/pg22234.txt',
  'date': '1879 copyright; 1880 edition',
  'date_evidence': 'Copyright 1879; title page 1880.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/22234/pg22234.txt'],
  'note': 'Snowy setting and clear action/consequence, but an adult retrospective about poverty, work and '
          'despair; less directly matched to child domestic dialogue.',
  'selections': [{'start': '^WHAT A SHOVEL DID\\.$',
                  'end': '^CLAMS\\.$',
                  'natural_end': True,
                  'sha256': '54a99effd6a912723207a097876c363b59be38f6a0bc3f815438524d4f181e8b',
                  'words': 2526}]},
 {'id': '24873',
  'title': 'Lucy Maud Montgomery Short Stories, 1896 to 1901',
  'author': 'L. M. Montgomery',
  'url': 'https://www.gutenberg.org/cache/epub/24873/pg24873.txt',
  'date': '1899 / 1901',
  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the ebook release date.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/24873/pg24873-images.html'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': 'A Christmas Inspiration',
                  'start': '^A\\ Christmas\\ Inspiration$',
                  'end': '^A\\ Christmas\\ Mistake$',
                  'sha256': 'd2de073e6e35af98abbc64285ad786af0ab0f1030e55029509943f840316e6ee',
                  'words': 2176,
                  'year': 1901,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Do not let this author dominate the supplementary sampling pool.',
                  'natural_end': True},
                 {'title': 'A Christmas Mistake',
                  'start': '^A\\ Christmas\\ Mistake$',
                  'end': '^A\\ Strayed\\ Allegiance$',
                  'sha256': '3895efec63f1d0dfabbb59d5ecd96d9fbc319e14b6d70ce197e70ed2eadc566d',
                  'words': 2833,
                  'year': 1899,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Do not let this author dominate the supplementary sampling pool.',
                  'natural_end': True}]},
 {'id': '24874',
  'title': 'Lucy Maud Montgomery Short Stories, 1902 to 1903',
  'author': 'L. M. Montgomery',
  'url': 'https://www.gutenberg.org/cache/epub/24874/pg24874.txt',
  'date': '1902 / 1903',
  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the ebook release date.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/24874/pg24874-images.html'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': "Aunt Cyrilla's Christmas Basket",
                  'start': "^Aunt\\ Cyrilla's\\ Christmas\\ Basket$",
                  'end': "^Davenport's\\ Story$",
                  'sha256': 'b0347c627874f4f559fda48829218b690051e0ef3ab2ef1fee4a4af5601aadef',
                  'words': 3159,
                  'year': 1903,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Do not let this author dominate the supplementary sampling pool.',
                  'natural_end': True},
                 {'title': "The Josephs' Christmas",
                  'start': "^The\\ Josephs'\\ Christmas$",
                  'end': '^The\\ Magical\\ Bond\\ of\\ the\\ Sea$',
                  'sha256': 'f935bfc53dd3efa7229c57363aade3b5bdb0eb0243a15f7fd6f023970263b284',
                  'words': 2102,
                  'year': 1902,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Do not let this author dominate the supplementary sampling pool.',
                  'natural_end': True},
                 {'title': "The Osbornes' Christmas",
                  'start': "^The\\ Osbornes'\\ Christmas$",
                  'end': '^The\\ Romance\\ of\\ Aunt\\ Beatrice$',
                  'sha256': '90ccb9ca81800b2d5238b5aa006eb262ef7f23473ccf7f6237dd0eb7f8d9d700',
                  'words': 1422,
                  'year': 1903,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Do not let this author dominate the supplementary sampling pool.',
                  'natural_end': True}]},
 {'id': '24876',
  'title': 'Lucy Maud Montgomery Short Stories, 1905 to 1906',
  'author': 'L. M. Montgomery',
  'url': 'https://www.gutenberg.org/cache/epub/24876/pg24876.txt',
  'date': '1905 / 1906',
  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the ebook release date.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/24876/pg24876-images.html'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': "Bertie's New Year",
                  'start': "^Bertie's\\ New\\ Year$",
                  'end': '^Between\\ the\\ Hill\\ and\\ the\\ Valley$',
                  'sha256': '389297b672f896648b6ec42693ef160109779f9119ee22380d76cad8b350739d',
                  'words': 2649,
                  'year': 1905,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Do not let this author dominate the supplementary sampling pool.',
                  'natural_end': True},
                 {'title': "Clorinda's Gifts",
                  'start': "^Clorinda's\\ Gifts$",
                  'end': "^Cyrilla's\\ Inspiration$",
                  'sha256': '7af89389c008281387b72eaa57e7c322e398a294d764b37f58fd5b070062a069',
                  'words': 1927,
                  'year': 1906,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Do not let this author dominate the supplementary sampling pool.',
                  'natural_end': True},
                 {'title': "Ida's New Year Cake",
                  'start': "^Ida's\\ New\\ Year\\ Cake$",
                  'end': '^In\\ the\\ Old\\ Valley$',
                  'sha256': '2b21f3e24c051a5b5ede2e64b032be83dcc8d041667594bace0f6363ccb64945',
                  'words': 2164,
                  'year': 1905,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Do not let this author dominate the supplementary sampling pool.',
                  'natural_end': True},
                 {'title': 'The Christmas Surprise at Enderly Road',
                  'start': '^The\\ Christmas\\ Surprise\\ at\\ Enderly\\ Road$',
                  'end': '^The\\ Dissipation\\ of\\ Miss\\ Ponsonby$',
                  'sha256': '8a5713dd94648c38c7561a7bb847af64031fde5a67a11c825213f1b997a039dd',
                  'words': 1748,
                  'year': 1905,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Do not let this author dominate the supplementary sampling pool.',
                  'natural_end': True},
                 {'title': "The Falsoms' Christmas Dinner",
                  'start': "^The\\ Falsoms'\\ Christmas\\ Dinner$",
                  'end': '^The\\ Fraser\\ Scholarship$',
                  'sha256': 'bee88cd02efbbebe46ee8fd3b97a0784935342e54e438cb770d377fe9440d359',
                  'words': 2822,
                  'year': 1906,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Do not let this author dominate the supplementary sampling pool.',
                  'natural_end': True},
                 {'title': 'The Light on the Big Dipper',
                  'start': '^The\\ Light\\ on\\ the\\ Big\\ Dipper$',
                  'end': '^The\\ Prodigal\\ Brother$',
                  'sha256': 'e588163e27cb446cdb7876566121812512a9fe74a6c3939f6a8272b826191f95',
                  'words': 2721,
                  'year': 1906,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Lighthouse rescue rather than domestic routine.',
                  'natural_end': True},
                 {'title': 'The Unforgotten One',
                  'start': '^The\\ Unforgotten\\ One$',
                  'end': '^The\\ Wooing\\ of\\ Bessy$',
                  'sha256': '403af9d5ba5a7145e450c5151ac3d31341fa35fb83c92662c485afc801148f5f',
                  'words': 2261,
                  'year': 1906,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Bereavement and a cemetery visit.',
                  'natural_end': True}]},
 {'id': '24878',
  'title': 'Lucy Maud Montgomery Short Stories, 1909 to 1922',
  'author': 'L. M. Montgomery',
  'url': 'https://www.gutenberg.org/cache/epub/24878/pg24878.txt',
  'date': '1909 / 1910',
  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the ebook release date.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/24878/pg24878-images.html'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': 'Christmas at Red Butte',
                  'start': '^Christmas\\ at\\ Red\\ Butte$',
                  'end': '^How\\ We\\ Went\\ to\\ the\\ Wedding$',
                  'sha256': '0529eb1006e1a37ecaaad0a2d0b4fccb65a7bf31f971404cfb6ed7ff13f35809',
                  'words': 2896,
                  'year': 1909,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Do not let this author dominate the supplementary sampling pool.',
                  'natural_end': True},
                 {'title': "Uncle Richard's New Year's Dinner",
                  'start': "^Uncle\\ Richard's\\ New\\ Year's\\ Dinner$",
                  'end': '^White\\ Magic$',
                  'sha256': 'd67ae1fd9e7edd004fa8c4ceb31d9e8e170a4dd3346dd681dd5cb530020f3592',
                  'words': 1515,
                  'year': 1910,
                  'date_evidence': 'Individual story year in the Gutenberg collection contents; not the '
                                   'ebook release date.',
                  'fit': 'Complete seasonal narrative with a concrete problem, actions and an explicit '
                         'outcome.',
                  'caveats': 'Period moralizing, occasional dialect and sentiment; several stories have '
                             'larger casts. Do not let this author dominate the supplementary sampling pool.',
                  'natural_end': True}]},
 {'id': '19766',
  'title': 'Young Lucretia and Other Stories',
  'author': 'Mary Eleanor Wilkins Freeman',
  'url': 'https://www.gutenberg.org/cache/epub/19766/pg19766.txt',
  'date': '1892',
  'date_evidence': 'Collection copyright 1892; transcribed title page 1893. Individual first appearances not '
                   'established.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/19766/pg19766-images.html'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': 'Young Lucretia',
                  'start': '^YOUNG\\ LUCRETIA\\n\\n',
                  'end': '^HOW\\ FIDELIA\\ WENT\\ TO\\ THE\\ STORE$',
                  'sha256': '6d1edf497889ead542ea24c4daa7adb9cf9d985f7e5fbd2c48039afb2519c8c8',
                  'words': 3808,
                  'year': 1892,
                  'date_evidence': 'Collection copyright 1892; transcribed title page 1893. Individual first '
                                   'appearances not established.',
                  'fit': 'Child/family problem with a clear resolution; gifts, a lost child, or missing '
                         'holiday food.',
                  'caveats': 'Frequent New England dialect. Thanksgiving selection is late-autumn domestic '
                             'fiction.',
                  'natural_end': True},
                 {'title': 'Where The Christmas-Tree Grew',
                  'start': '^WHERE\\ THE\\ CHRISTMAS\\-TREE\\ GREW\\n\\n',
                  'end': "^WHERE\\ SARAH\\ JANE'S\\ DOLL\\ WENT$",
                  'sha256': 'c2211c844fbf6764407e61a558cde5d88fc0e401d88111a7e855660c42d28b2b',
                  'words': 2933,
                  'year': 1892,
                  'date_evidence': 'Collection copyright 1892; transcribed title page 1893. Individual first '
                                   'appearances not established.',
                  'fit': 'Child/family problem with a clear resolution; gifts, a lost child, or missing '
                         'holiday food.',
                  'caveats': 'Frequent New England dialect. Thanksgiving selection is late-autumn domestic '
                             'fiction.',
                  'natural_end': True},
                 {'title': 'Ann Mary Her Two Thanksgivings',
                  'start': '^ANN\\ MARY\\ HER\\ TWO\\ THANKSGIVINGS\\n\\n',
                  'end': "^ANN\\ LIZY'S\\ PATCHWORK$",
                  'sha256': 'cf183739245fd09e3d04ff3c6d7075a16bd4b65208c5e3a9fd40b396362b98e4',
                  'words': 5525,
                  'year': 1892,
                  'date_evidence': 'Collection copyright 1892; transcribed title page 1893. Individual first '
                                   'appearances not established.',
                  'fit': 'Child/family problem with a clear resolution; gifts, a lost child, or missing '
                         'holiday food.',
                  'caveats': 'Frequent New England dialect. Thanksgiving selection is late-autumn domestic '
                             'fiction.',
                  'natural_end': True}]},
 {'id': '17888',
  'title': 'Comfort Pease and her Gold Ring',
  'author': 'Mary Eleanor Wilkins Freeman',
  'url': 'https://www.gutenberg.org/cache/epub/17888/pg17888.txt',
  'date': '1895',
  'date_evidence': 'Transcribed title page MDCCCXCV (1895).',
  'date_sources': ['https://www.gutenberg.org/cache/epub/17888/pg17888-images.html'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': 'Comfort Pease and her Gold Ring',
                  'start': '^One of the first things which Comfort remembered',
                  'end': '\\Z',
                  'sha256': '15f1bbf75d603b207bfc67d24f7df90a3298364658b01113c18e145445f515dd',
                  'words': 11043,
                  'year': 1895,
                  'date_evidence': 'Transcribed title page MDCCCXCV (1895).',
                  'fit': 'One sustained causal plot: a child borrows, loses, searches for, and confesses '
                         'about a ring, in snowy school/home settings.',
                  'caveats': 'Longer than a short scene; dialect and explicit moral instruction. Final '
                             'school coda is three months later.',
                  'natural_end': True}]},
 {'id': '17530',
  'title': "Maida's Little Shop",
  'author': 'Inez Haynes Gillmore',
  'url': 'https://www.gutenberg.org/cache/epub/17530/pg17530.txt',
  'date': '1909',
  'date_evidence': 'Copyright 1909 printed in the transcribed edition; later-series advertising excluded.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/17530/pg17530-images.html'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': "Maida's Little Shop, XII-XIV",
                  'start': '^ CHAPTER XII: THE FIRST SNOW$',
                  'end': '^Yes, Maida did come back\\.',
                  'sha256': '1619035044a94059d8a1d8b382901e8207f6825afbf5a0f1df13804cc8a3c1e9',
                  'words': 9127,
                  'year': 1909,
                  'date_evidence': 'Copyright 1909 printed in the transcribed edition; later-series '
                                   'advertising excluded.',
                  'fit': 'The First Snow, The Fair, Christmas Happenings: snow play, making and selling '
                         'gifts, and a Christmas reunion.',
                  'caveats': 'Prior cast context; Granny uses dialect. Stops after the narrative farewell '
                             'and before the sequel advertisement.',
                  'natural_end': True}]},
 {'id': '34378',
  'title': 'Hans Brinker; Or, The Silver Skates',
  'author': 'Mary Mapes Dodge',
  'url': 'https://www.gutenberg.org/cache/epub/34378/pg34378.txt',
  'date': '1865',
  'date_evidence': 'First publication 1865 in the Kalamazoo Public Library catalog; Gutenberg preface/letter '
                   'itself dated 1873. Exact printing date of the undated Cooke-illustrated transcription '
                   'not established.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/34378/pg34378-images.html',
                   'https://www.kpl.gov/catalog/item/?i=ent%3A%2F%2FERC_215_8682%2F0%2F215_8682%3AHOOPLA%3A13249740'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': 'Hans Brinker, I: Hans and Gretel',
                  'start': '^I\\n\\nHANS AND GRETEL$',
                  'end': '^II$',
                  'sha256': 'a243865e7e00945eba07e1dab343fc5a6465b76436ba55e8160d9f907601e239',
                  'words': 1245,
                  'year': 1865,
                  'date_evidence': 'First publication 1865 in the Kalamazoo Public Library catalog; '
                                   'Gutenberg preface/letter itself dated 1873. Exact printing date of the '
                                   'undated Cooke-illustrated transcription not established.',
                  'fit': 'December skating, obtaining skates, family care and the Saint Nicholas '
                         'celebration.',
                  'caveats': 'Some authorial addresses, exposition and archaic pronouns remain. '
                             'Historical/geographical chapters omitted.',
                  'natural_end': False},
                 {'title': 'Hans Brinker, III-VII',
                  'start': '^III\\n\\nTHE SILVER SKATES$',
                  'end': '^VIII$',
                  'sha256': '8f313ade46f1a7c0b4cb3668ed943d69796e73b88beefdf0ca1bc78692197298',
                  'words': 7966,
                  'year': 1865,
                  'date_evidence': 'First publication 1865 in the Kalamazoo Public Library catalog; '
                                   'Gutenberg preface/letter itself dated 1873. Exact printing date of the '
                                   'undated Cooke-illustrated transcription not established.',
                  'fit': 'December skating, obtaining skates, family care and the Saint Nicholas '
                         'celebration.',
                  'caveats': 'Some authorial addresses, exposition and archaic pronouns remain. '
                             'Historical/geographical chapters omitted.',
                  'natural_end': False},
                 {'title': 'Hans Brinker, IX: The Festival of Saint Nicholas',
                  'start': '^IX\\n\\nTHE FESTIVAL OF SAINT NICHOLAS$',
                  'end': '^X$',
                  'sha256': 'a0def05504d5a16dd64d422a5593b5967802f7b43e4d08b25b06bdacaf4d3492',
                  'words': 2569,
                  'year': 1865,
                  'date_evidence': 'First publication 1865 in the Kalamazoo Public Library catalog; '
                                   'Gutenberg preface/letter itself dated 1873. Exact printing date of the '
                                   'undated Cooke-illustrated transcription not established.',
                  'fit': 'December skating, obtaining skates, family care and the Saint Nicholas '
                         'celebration.',
                  'caveats': 'Some authorial addresses, exposition and archaic pronouns remain. '
                             'Historical/geographical chapters omitted.',
                  'natural_end': False}]},
 {'id': '76681',
  'title': "Liliecrona's home",
  'author': 'Selma Lagerlöf',
  'url': 'https://www.gutenberg.org/cache/epub/76681/pg76681.txt',
  'date': '1914',
  'date_evidence': 'This English translation by Anna Barwell has a New York title page dated MCMXIV (1914).',
  'date_sources': ['https://www.gutenberg.org/cache/epub/76681/pg76681-images.html'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': "Liliecrona's Home, I-II",
                  'start': '^CHAPTER I$',
                  'end': '^CHAPTER III$',
                  'sha256': 'c88873d273fcf879bc8234c77d766e6affefcdc18b019e39cf69b3fcdb45baaf',
                  'words': 6699,
                  'year': 1914,
                  'date_evidence': 'This English translation by Anna Barwell has a New York title page dated '
                                   'MCMXIV (1914).',
                  'fit': 'Christmas storm, seeking shelter, and a child entering a household; next-morning '
                         'spinning-room scene.',
                  'caveats': 'Translated prose and larger novel setup; threatened exposure and an oppressive '
                             'household. Not a complete novel arc.',
                  'natural_end': False}]},
 {'id': '56583',
  'title': 'Patty in the City',
  'author': 'Carolyn Wells',
  'url': 'https://www.gutenberg.org/cache/epub/56583/pg56583.txt',
  'date': '1905',
  'date_evidence': 'Copyright 1905 in the transcribed front matter.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/56583/pg56583-images.html'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': 'Patty in the City, XXIII: Christmas',
                  'start': '^ CHAPTER XXIII$',
                  'end': '^ \\* \\* \\* \\* \\*\\n\\n “_The Books you like to read',
                  'sha256': '9e6bc09d5a2f863e78106acc6f83ef68a0001f1619deaef26fb25788427ecf14',
                  'words': 2104,
                  'year': 1905,
                  'date_evidence': 'Copyright 1905 in the transcribed front matter.',
                  'fit': 'Complete final chapter: family Christmas visits, presents and conversation.',
                  'caveats': 'Large established cast and romance references. Publisher lists excluded; same '
                             'series as an original training book.',
                  'natural_end': True}]},
 {'id': '316',
  'title': 'The Golden Road',
  'author': 'L. M. Montgomery',
  'url': 'https://www.gutenberg.org/cache/epub/316/pg316.txt',
  'date': '1913',
  'date_evidence': 'A separate Gutenberg Canada transcription reproduces copyright 1913 and First '
                   'Impression, August 1913.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/316/pg316-images.html',
                   'https://gutenbergcanada.ca/ebooks/montgomery-golden/montgomery-golden-00-h-dir/montgomery-golden-00-h.html'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': 'The Golden Road, III-IV',
                  'start': '^CHAPTER III\\. THE CHRISTMAS HARP$',
                  'end': '^CHAPTER V\\.',
                  'sha256': 'e400f93d211e63356549b9cc6b00ed7890e5153981aa081ba2e660132e7f16f3',
                  'words': 6331,
                  'year': 1913,
                  'date_evidence': 'A separate Gutenberg Canada transcription reproduces copyright 1913 and '
                                   'First Impression, August 1913.',
                  'fit': 'Christmas preparations and presents followed by snowy New Year resolutions.',
                  'caveats': 'Many children, an embedded legend, verse and reflective narration; use after '
                             'the simpler complete stories.',
                  'natural_end': False}]},
 {'id': '544',
  'title': "Anne's House of Dreams",
  'author': 'L. M. Montgomery',
  'url': 'https://www.gutenberg.org/cache/epub/544/pg544.txt',
  'date': '1917',
  'date_evidence': 'University of Pennsylvania Online Books Page lists the work in a 1917 edition; the '
                   'Montgomery bibliography documents advertisements and reviews from August 1917.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/544/pg544-images.html',
                   'https://onlinebooks.library.upenn.edu/webbin/book/lookupname?key=Montgomery%2C+L.+M.+%28Lucy+Maud%29%2C+1874-1942',
                   'https://lmmonline.org/bibliography/magazine-items-1/'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': "Anne's House of Dreams, 15-17",
                  'start': '^CHAPTER 15$',
                  'end': '^CHAPTER 18$',
                  'sha256': '6735357051600b447ee26277f9a4a583be495b36101f357a192e1a87bd7962bf',
                  'words': 6339,
                  'year': 1917,
                  'date_evidence': 'University of Pennsylvania Online Books Page lists the work in a 1917 '
                                   'edition; the Montgomery bibliography documents advertisements and '
                                   'reviews from August 1917.',
                  'fit': "Christmas at Four Winds, New Year's Eve at the Light, A Four Winds Winter: visits, "
                         'meals and household conversations.',
                  'caveats': 'Adult cast, embedded reminiscences and continuing-novel relationships; more '
                             "demanding than the children's short stories.",
                  'natural_end': False}]},
 {'id': '45',
  'title': 'Anne of Green Gables',
  'author': 'L. M. Montgomery',
  'url': 'https://www.gutenberg.org/cache/epub/45/pg45.txt',
  'date': '1908',
  'date_evidence': 'Separate illustrated Gutenberg edition 64365 reproduces copyright 1908 and '
                   'August/September 1908 impressions.',
  'date_sources': ['https://www.gutenberg.org/cache/epub/45/pg45-images.html',
                   'https://www.gutenberg.org/cache/epub/64365/pg64365-images.html'],
  'note': 'Round 3: original prose; openings, middle passages, endings and seasonal context reviewed. '
          'Individual story dates and caveats appear below.',
  'selections': [{'title': 'Anne of Green Gables, XVIII-XIX',
                  'start': '^CHAPTER XVIII\\. Anne to the Rescue$',
                  'end': '^CHAPTER XX\\.',
                  'sha256': '3d5036434ce9d6c8f5b279e4c4c27a1681691bc1dc2c2eeee323f46b32b8d95d',
                  'words': 7434,
                  'year': 1908,
                  'date_evidence': 'Separate illustrated Gutenberg edition 64365 reproduces copyright 1908 '
                                   'and August/September 1908 impressions.',
                  'fit': 'Winter illness/rescue and a birthday visit; Christmas concert preparations and '
                         "Matthew's gift of a dress.",
                  'caveats': 'Prior novel context, longer dialogue and a historical medical scene; keep one '
                             'copy of the puffed-sleeves chapter.',
                  'natural_end': False},
                 {'title': 'Anne of Green Gables, XXIV-XXV',
                  'start': '^CHAPTER XXIV\\. Miss Stacy and Her Pupils Get Up a Concert$',
                  'end': '^CHAPTER XXVI\\.',
                  'sha256': '1f4acec3902158db742516de9e7b11a08c0f4fc2cb854fe9ba21e10cb4c3bd82',
                  'words': 4991,
                  'year': 1908,
                  'date_evidence': 'Separate illustrated Gutenberg edition 64365 reproduces copyright 1908 '
                                   'and August/September 1908 impressions.',
                  'fit': 'Winter illness/rescue and a birthday visit; Christmas concert preparations and '
                         "Matthew's gift of a dress.",
                  'caveats': 'Prior novel context, longer dialogue and a historical medical scene; keep one '
                             'copy of the puffed-sleeves chapter.',
                  'natural_end': False}]}]

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_sources(path=None):
    """Use embedded URLs, or parse an explicit file without executing its Python."""
    selected = sources
    if path is not None:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "sources" for t in node.targets
            ):
                selected = ast.literal_eval(node.value)
                break
        else:
            raise ValueError("No literal sources list found")
    if not isinstance(selected, list) or not all(isinstance(s, str) for s in selected):
        raise ValueError("sources must be a literal list of URLs")
    for url in selected:
        if not re.fullmatch(r"https://www\.gutenberg\.org/cache/epub/(\d+)/pg\1\.txt", url):
            raise ValueError(f"Unexpected corpus URL: {url}")
        if url not in sources:
            raise ValueError(f"Source is outside the approved Christmas corpus: {url}")
    return list(dict.fromkeys(selected))


def source_identity(path=None):
    encoded = json.dumps(read_sources(path), ensure_ascii=False, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    accepted = {fingerprint}
    if path is not None:
        accepted.add(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    if fingerprint == ORIGINAL_SOURCES_SHA256:
        # Leave legacy reports untouched so existing checkpoint signatures still match.
        accepted.add(ORIGINAL_CORPUS_SHA256)
    return fingerprint, accepted


def clean_book(raw):
    """Conservative Gutenberg cleanup; keep prose, dialogue, and verse layout."""
    text = unicodedata.normalize("NFC", raw.replace("\r\n", "\n").replace("\r", "\n"))
    if "<html" in text[:1000].lower() or "<!doctype html" in text[:1000].lower():
        raise ValueError("Received HTML instead of a book")
    start = re.search(r"^\s*\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*[^\n]*\n", text, re.M | re.I)
    end = re.search(r"^\s*\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.M | re.I)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Missing or invalid Gutenberg boundary markers; refusing unstripped text")
    header, body = text[:start.start()], text[start.end():end.start()]
    metadata = {}
    for field in ("Title", "Author", "Language"):
        match = re.search(rf"^{field}:\s*(.+(?:\n[ \t]+[^\n]+)*)", header, re.M)
        metadata[field.lower()] = re.sub(r"\s+", " ", match.group(1)).strip() if match else "Unknown"
    if metadata["language"].lower() != "english":
        raise ValueError(f"Unexpected language: {metadata['language']}")
    body = body.replace("\ufeff", "").replace("\u00ad", "").replace("\xa0", " ")
    body = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", body)
    body = re.sub(r"(?im)^\s*(?:Produced by|Transcribed by|This eBook was produced by)[^\n]*(?:\n[^\n]+)*\n\s*\n", "\n\n", body)
    body = re.sub(r"(?is)\[(?:Illustration|Transcriber[’']?s? note)[^\]]*\]", "", body)
    body = re.sub(r"(?im)^\s*\[?(?:Page|Pg\.)\s+\d+\]?\s*$", "", body)
    paragraphs = []
    for block in re.split(r"\n\s*\n", body):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        # Prose is usually wrapped near 70 columns. Retain short-line verse.
        prose = len(lines) > 1 and sum(len(x.strip()) >= 55 for x in lines[:-1]) / (len(lines)-1) >= .65
        prose = prose and not all(x.startswith("    ") for x in lines)
        paragraphs.append(" ".join(x.strip() for x in lines) if prose else "\n".join(lines))
    body = re.sub(r"[ \t]+", " ", "\n\n".join(paragraphs)).strip()
    if len(body.split()) < 300:
        raise ValueError("Book has fewer than 300 words after cleaning")
    if re.search(r"END OF THE PROJECT GUTENBERG|FULL PROJECT GUTENBERG LICENSE", body, re.I):
        raise ValueError("Boilerplate survived cleanup")
    return metadata, body


def download_book(url, cache, offline=False):
    import requests
    book_id = url.split("/")[-2]
    path = Path(cache) / "raw" / f"{book_id}.txt"
    if path.exists():
        # Preserve CRLF so source hashes agree between downloads and cache hits.
        with path.open(encoding="utf-8", newline="") as handle:
            return handle.read()
    if offline:
        raise FileNotFoundError(f"No cached download: {path}")
    error = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 60), headers={"User-Agent": "christmasmodel-corpus/1.0"})
            response.raise_for_status()
            raw = response.content.decode("utf-8-sig", errors="strict")
            clean_book(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(raw, encoding="utf-8")
            tmp.replace(path)
            time.sleep(.25)  # Sequential, cached, modest requests to Gutenberg.
            return raw
        except (requests.RequestException, UnicodeError, ValueError) as exc:
            error = exc
            if isinstance(exc, requests.HTTPError) and exc.response.status_code == 404:
                break
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{url}: {error}")


def fingerprints(text):
    """Content-sampled 13-word shingles, independent of line/paragraph wrapping."""
    words = re.findall(r"\w+", text.casefold())
    result = set()
    for i in range(len(words) - 12):
        value = int.from_bytes(hashlib.blake2b(" ".join(words[i:i+13]).encode(), digest_size=8).digest(), "little")
        if value % 32 == 0:
            result.add(value)
    return result


def group_books(books):
    """Remove near-identical editions; group overlapping anthologies for splitting."""
    kept, signatures, duplicates = [], [], []
    for book in books:
        sig = fingerprints(book["text"])
        duplicate = next((i for i, old in enumerate(signatures)
                          if min(len(sig), len(old)) and len(sig & old) / max(len(sig), len(old)) >= .85), None)
        if duplicate is not None:
            duplicates.append({"id": book["id"], "duplicate_of": kept[duplicate]["id"]})
        else:
            kept.append(book)
            signatures.append(sig)
    parents = list(range(len(kept)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    overlaps = []
    for i in range(len(kept)):
        for j in range(i):
            shared = len(signatures[i] & signatures[j])
            containment = shared / max(1, min(len(signatures[i]), len(signatures[j])))
            if shared >= 12 and containment >= .05:
                parents[find(i)] = find(j)
                overlaps.append({"ids": [kept[j]["id"], kept[i]["id"]], "containment": round(containment, 4)})
    groups = defaultdict(list)
    for i, book in enumerate(kept):
        groups[find(i)].append(book)
    return list(groups.values()), duplicates, overlaps


def split_groups(groups, seed):
    """Choose whole overlap groups near 90/5/5 by word count, not random windows."""
    if len(groups) < 3:
        raise ValueError("Need at least three independent book groups for train/validation/test")
    sizes = [sum(b["words"] for b in g) for g in groups]
    total = sum(sizes)
    rng, best = random.Random(seed), None
    for _ in range(1000):
        order = list(range(len(groups)))
        rng.shuffle(order)
        assigned = {"train": [], "validation": [], "test": []}
        remaining = order[:]
        for split in ("validation", "test"):
            count = 0
            while remaining and (not assigned[split] or count < total * .05):
                if len(remaining) <= (1 if split == "test" else 2) and assigned[split]:
                    break
                i = remaining.pop()
                assigned[split].append(i)
                count += sizes[i]
        assigned["train"] = remaining
        if not remaining:
            continue
        score = sum(abs(sum(sizes[i] for i in assigned[s])/total - .05) for s in ("validation", "test"))
        if best is None or score < best[0]:
            best = (score, assigned)
    if best is None:
        raise ValueError("Unable to form nonempty grouped splits")
    return {s: [b for i in ids for b in groups[i]] for s, ids in best[1].items()}


def prepare(args):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2TokenizerFast
    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    source_hash, accepted_hashes = source_identity(args.corpus)
    if (data / "report.json").exists() and not args.rebuild:
        report = json.loads((data / "report.json").read_text())
        if report["source_sha256"] not in accepted_hashes or (
            report["seed"], report["requested_vocab_size"], report["cleaning_version"]
        ) != (args.seed, args.vocab_size, CLEANING_VERSION):
            raise ValueError("Preparation settings/corpus changed; use a new --data-dir or --rebuild")
        if report["downloaded"] / report["sources"] < args.min_success_rate:
            raise ValueError("Cached corpus misses the requested success rate; retry with --rebuild")
        required = ["train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json", "tokenizer/tokenizer.json"]
        if any(not (data / name).is_file() for name in required):
            raise ValueError("Prepared corpus is incomplete; restore it or use --rebuild")
        print(f"Using prepared corpus: {data / 'report.json'}", flush=True)
        return report
    sources = read_sources(args.corpus)
    books, failures = [], []
    for i, url in enumerate(sources):
        try:
            raw = download_book(url, data, args.offline)
            metadata, text = clean_book(raw)
            books.append(dict(id=url.split("/")[-2], url=url, text=text, words=len(text.split()),
                              sha256=hashlib.sha256(raw.encode()).hexdigest(), **metadata))
            print(f"[{i+1}/{len(sources)}] {metadata['title'][:70]}: {len(text.split()):,} words", flush=True)
        except (RuntimeError, ValueError, OSError) as exc:
            failures.append({"url": url, "error": str(exc)})
            print(f"SKIPPED: {exc}", flush=True)
    write_json(data / "download_failures.json", failures)
    if len(books) < 3 or len(books) / len(sources) < args.min_success_rate:
        raise RuntimeError("Too few valid books; inspect download_failures.json and retry. No training performed.")
    print("Detecting duplicate editions and overlapping stories...", flush=True)
    groups, duplicates, overlaps = group_books(books)
    splits = split_groups(groups, args.seed)
    tokenizer_dir = data / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    bpe = ByteLevelBPETokenizer(add_prefix_space=False)
    bpe.train_from_iterator((b["text"] for b in splits["train"]), vocab_size=args.vocab_size,
                            min_frequency=3, special_tokens=[EOT, PAD], show_progress=True)
    bpe.save_model(str(tokenizer_dir))
    tokenizer = GPT2TokenizerFast(vocab_file=str(tokenizer_dir / "vocab.json"),
                                 merges_file=str(tokenizer_dir / "merges.txt"),
                                 bos_token=EOT, eos_token=EOT, unk_token=EOT, pad_token=PAD,
                                 add_prefix_space=False, add_bos_token=False, model_max_length=1_000_000)
    tokenizer.save_pretrained(tokenizer_dir)
    stats, manifest = {}, []
    for split, subset in splits.items():
        token_count = 0
        with (data / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for book in subset:
                ids = tokenizer.encode(book["text"], add_special_tokens=False)
                token_count += len(ids) + 1
                handle.write(json.dumps({"id": book["id"], "ids": [tokenizer.eos_token_id] + ids + [tokenizer.eos_token_id]}) + "\n")
                clean_path = data / "clean" / f"{book['id']}.txt"
                clean_path.parent.mkdir(exist_ok=True)
                clean_path.write_text(book["text"], encoding="utf-8")
                manifest.append({k: v for k, v in book.items() if k != "text"} | {"split": split, "tokens": len(ids)})
        stats[split] = {"books": len(subset), "words": sum(b["words"] for b in subset), "target_tokens": token_count}
    report = {"cleaning_version": CLEANING_VERSION, "source_sha256": source_hash,
              "seed": args.seed, "requested_vocab_size": args.vocab_size, "vocab_size": len(tokenizer),
              "sources": len(sources), "downloaded": len(books), "failures": failures,
              "duplicates": duplicates, "overlaps": overlaps, "independent_groups": len(groups), "splits": stats}
    write_json(data / "manifest.json", manifest)
    write_json(data / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


class BookWindows:
    """Fixed windows score every next-token target once, including book tails.

    Windows never cross books. Real EOS labels stay supervised; only padding is
    -100. Optional training offsets move interior windows each epoch, keeping
    row lengths and edge windows fixed; then coverage is sampled, not exhaustive.
    """
    def __init__(self, path, block_size, limit=None, random_offsets=False, seed=42):
        import numpy as np
        self.books, self.book_ids, self.windows = [], [], []
        self.block_size = self.target_block_size = block_size
        self.random_offsets, self.seed = random_offsets, seed
        for line in Path(path).read_text().splitlines():
            book = json.loads(line)
            ids = np.asarray(book["ids"], dtype=np.int32)
            index = len(self.books)
            self.books.append(ids)
            self.book_ids.append(str(book["id"]))
            self.windows.extend((index, start) for start in range(0, len(ids)-1, block_size-1))
        if limit:
            # Spread smoke/evaluation subsets across the corpus, not just book 1.
            indices = np.linspace(0, len(self.windows)-1, min(limit, len(self.windows)), dtype=int)
            self.windows = [self.windows[i] for i in indices]
        if not self.windows:
            raise ValueError(f"Empty token dataset: {path}")
        self.set_training_epoch(0)

    def set_training_epoch(self, epoch):
        """Deterministic per-book shifts; independent of global training RNG."""
        self.epoch = int(epoch)
        span = self.target_block_size-1
        self.offsets = [int.from_bytes(hashlib.sha256(
            f"{self.seed}:{self.epoch}:{book_id}".encode()).digest()[:8], "big") % (2*span+1)-span
            for book_id in self.book_ids] if self.random_offsets else [0]*len(self.books)

    def __len__(self):
        return len(self.windows)

    def target_window(self, index):
        book, start = self.windows[index]
        if self.random_offsets and start > 0 and start+self.target_block_size < len(self.books[book]):
            # Keep the original first/last windows (and all EOS occurrences).
            # Interior windows stay contiguous within their original book.
            start = max(1, min(start+self.offsets[book], len(self.books[book])-self.target_block_size-1))
        return book, start, min(start+self.target_block_size, len(self.books[book]))

    def scored_targets(self, index):
        _, start, end = self.target_window(index)
        return end-start-1

    def __getitem__(self, index):
        book, start, end = self.target_window(index)
        return {"input_ids": self.books[book][start:end].tolist()}


class ContextBookWindows(BookWindows):
    """Predict a target span with additional preceding text from the SAME document.

    At context 512 and lookback 128, each full row predicts 383 targets. The
    preceding 129 tokens (where available) are visible but their labels masked.
    Fixed windows cover each original target exactly once. Shifted training
    windows retain the prior sampling policy and can repeat/omit interior targets
    in a pass; prepending context itself never adds duplicate loss targets.
    """
    def __init__(self, path, block_size, limit=None, random_offsets=False, seed=42, lookback=128):
        if not 0 <= lookback <= block_size-2:
            raise ValueError("Lookback must leave at least one causal target")
        super().__init__(path, block_size-lookback, limit, random_offsets, seed)
        self.block_size, self.lookback = block_size, lookback

    def __getitem__(self, index):
        book, start, end = self.target_window(index)
        begin = max(0, start-self.lookback)
        ids = self.books[book][begin:end].tolist()
        labels = ids.copy()
        labels[:start-begin+1] = [-100]*(start-begin+1)
        return {"input_ids": ids, "labels": labels}


def window_epoch_callback(dataset):
    """Use Trainer's restored epoch, including when its loader skips batches."""
    from transformers import TrainerCallback

    class WindowEpochCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            # Do not expose Dataset.set_epoch: Accelerate's resumed skip loader
            # can initially call that hook with zero even in a later epoch.
            dataset.set_training_epoch(int(state.epoch or 0))
            if dataset.random_offsets:
                print(f"Training window offsets: epoch {dataset.epoch}; "
                      "book boundaries and target counts preserved.", flush=True)

    return WindowEpochCallback()


def extend_gpt2_context(source, positions):
    """Retain every old position exactly; initialize new positions by tiling them.

    Rebuild the causal attention buffers too. New positions REQUIRE training;
    tiling is an initialization, not a claim of zero-shot context extension.
    """
    import copy
    import torch
    from transformers import GPT2LMHeadModel
    old = source.config.n_positions
    if positions < old:
        raise ValueError("Cannot shrink a trained context")
    if positions == old:
        return source, {"old_positions": old, "new_positions": positions, "changed": False}
    config = copy.deepcopy(source.config)
    config.n_positions = config.n_ctx = positions
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)  # CPU initialization; leave GPU RNGs untouched.
        grown = GPT2LMHeadModel(config).to(device=source.device, dtype=source.dtype)
    state = source.state_dict()
    state["transformer.wpe.weight"] = source.transformer.wpe.weight.detach().repeat(
        math.ceil(positions/old), 1)[:positions].clone()
    grown.load_state_dict(state, strict=True)
    grown.tie_weights()
    grown.generation_config = copy.deepcopy(source.generation_config)
    grown.train(source.training)
    return grown, {"old_positions": old, "new_positions": positions, "changed": True,
                   "initialization": "old positions copied exactly; added positions tiled, then trained"}


def passage_simplicity(text):
    """Bounded surface proxy, not a semantic quality label or a text filter."""
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return .5
    sentences = max(1, len(re.findall(r'[.!?](?:[\s\"”]|$)', text)))
    names = len(set(re.findall(r'(?<=[a-z,;] )\b[A-Z][a-z]{2,}\b', text)))
    long_words = sum(len(w) > 10 for w in words)/len(words)
    value = 22/max(8, len(words)/sentences)/(1+names/8+4*long_words)
    return max(.5, min(1.5, value))


def prose_sentence_ends(text):
    """Conservative punctuation boundaries: titles/initials are not sentences."""
    ends = []
    abbreviations = {'mr', 'mrs', 'ms', 'dr', 'st', 'rev', 'prof', 'jr', 'sr', 'etc', 'vs'}
    for match in re.finditer(r'[.!?][\"\u201d\u2019\x27)]*(?=\s|$)', text):
        word = re.search(r'([A-Za-z]+)$', text[max(0, match.start()-80):match.start()])
        if text[match.start()] == '.' and word and (word[1].lower() in abbreviations or len(word[1]) == 1):
            continue
        ends.append(match.end())
    return ends


def continuity_corruptions(prefix, target, person_names=None):
    """Corpus-derived negatives, not human labels or a semantic truth oracle."""
    alternatives = []
    # Names must occur inside sentences in the prefix, avoiding ordinary sentence starters.
    full_names = list(re.finditer(r'\b[A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]{2,}){0,2}\b', prefix))
    names = sorted({m.group() for m in full_names if re.search(r'[a-z,;] $',prefix[:m.start()])} -
                   {'Christmas', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                    'Saturday', 'December', 'January', 'February', 'God', 'Lord'})
    if person_names is not None:
        names = [name for name in names if name in person_names]
    for name in names:
        mentions = [m for m in re.finditer(r'\b[A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]{2,}){0,2}\b',target)
                    if m.group()==name]
        if mentions:
            other = next((n for n in names if n != name), None)
            if other:
                mention=mentions[0]
                alternatives.append(('entity', target[:mention.start()]+other+target[mention.end():]))
                break
    # Require an auxiliary construction: do not turn lexical "had a gift" into malformed prose.
    match = re.search(r'\b(was|were|is|are|could|would|should|will|can)( not)?\b', target, re.I)
    if match:
        replacement = match[1]+('' if match[2] else ' not')
        variant = 'polarity:remove_not' if match[2] else 'polarity:add_not'
        alternatives.append((variant, target[:match.start()]+replacement+target[match.end():]))
    boundaries = prose_sentence_ends(target)
    if len(boundaries) >= 2:
        first, second = boundaries[:2]
        # Keep punctuation/words; alter only order. Exclude ambiguous identical sentences.
        a, b = target[:first].strip(), target[first:second].strip()
        if a != b and min(len(a.split()), len(b.split())) >= 5 and not any(quote in prefix+target for quote in ('"', '\u201c', '\u201d')):
            leading = target[:len(target)-len(target.lstrip())]
            alternatives.append(('order', leading+b+' '+a+target[second:]))
    return [(kind, text) for kind, text in alternatives if text != target]


def build_continuity_pairs(dataset, tokenizer, per_book=128, seed=42):
    """Freeze exact sentence-bounded positives from ONLY the supplied book split.

    Prefixes contain 48..160 original tokens; targets contain 24..96 original
    tokens and end at a sentence boundary. No positive text is rewritten.
    """
    from bisect import bisect_right
    records = []
    for book_id, array in zip(dataset.book_ids, dataset.books):
        ids = array.tolist()
        if ids and ids[0] == tokenizer.eos_token_id:
            ids = ids[1:]
        if ids and ids[-1] == tokenizer.eos_token_id:
            ids = ids[:-1]
        text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        verbs = r'(?:said|asked|answered|replied|whispered|exclaimed|cried)'
        name_pattern=r'([A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]{2,}){0,2})'
        person_names = set(re.findall(r'\b'+name_pattern+r'\s+'+verbs+r'\b', text))
        person_names.update(re.findall(r'\b'+verbs+r'\s+'+name_pattern+r'\b', text))
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        if encoded['input_ids'] != ids:
            raise ValueError(f'Continuity source does not round-trip: {book_id}')
        ends = [span[1] for span in encoded['offset_mapping']]
        boundaries = sorted(set([0]+[bisect_right(ends, end) for end in prose_sentence_ends(text)]))
        rng = random.Random(f'{seed}:{book_id}:continuity-v1')
        indices = list(range(1, len(boundaries)-1))
        rng.shuffle(indices)
        accepted = 0
        for j in indices:
            start = boundaries[j]
            beginnings = [b for b in boundaries[max(0, j-8):j] if 48 <= start-b <= 160]
            endings = [b for b in boundaries[j+1:j+5] if 24 <= b-start <= 96]
            if not beginnings or not endings:
                continue
            begin = min(beginnings, key=lambda b: abs(start-b-112))
            end = min(endings, key=lambda b: abs(b-start-64))
            prefix_ids, chosen = ids[begin:start], ids[start:end]
            prefix = tokenizer.decode(prefix_ids, clean_up_tokenization_spaces=False)
            target = tokenizer.decode(chosen, clean_up_tokenization_spaces=False)
            if re.search(r'^\s*(?:CHAPTER|BOOK|PART)\b|^[A-Z][A-Z ]{12,}$', prefix+target, re.M):
                continue
            alternatives = continuity_corruptions(prefix, target, person_names)
            # Balance available corruption types rather than always favoring negation.
            rng.shuffle(alternatives)
            for kind, negative in alternatives:
                rejected = tokenizer.encode(negative, add_special_tokens=False)
                common = next((i for i, (a, b) in enumerate(zip(chosen, rejected)) if a != b),
                              min(len(chosen), len(rejected)))
                if common >= min(len(chosen), len(rejected)) or len(prefix_ids)+len(rejected) > 288:
                    continue
                record = {'book_id': book_id, 'start_token': begin+1, 'target_start_token': start+1,
                          'end_token': end+1, 'prefix_ids': prefix_ids, 'chosen_ids': chosen,
                          'rejected_ids': rejected, 'difference_start': common,
                          'kind': kind.split(':')[0], 'variant': kind}
                record['pair_id'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:20]
                records.append(record)
                accepted += 1
                break
            if accepted >= per_book:
                break
    # Otherwise insertion negatives dominate and reward globally suppressing "not".
    polarities = {v: [r for r in records if r['variant'] == v]
                  for v in ('polarity:add_not', 'polarity:remove_not')}
    balanced_count = min(map(len, polarities.values()))
    kept = {r['pair_id'] for group in polarities.values() for r in group[:balanced_count]}
    return [r for r in records if r['kind'] != 'polarity' or r['pair_id'] in kept]


def continuity_preference_loss(model, pairs, pad_id, beta=2., margin=.2):
    """Rank changed spans; separately anchor ALL original continuation tokens.

    Scoring at most 32 tokens after the first difference avoids diluting a
    changed name/event across an otherwise identical long passage. No reference
    model, external weights, generated positive story, or auxiliary export head.
    """
    import torch
    import torch.nn.functional as F
    if not pairs:
        zero = next(model.parameters()).new_zeros(())
        return zero, zero, {'pairs': 0, 'wins': 0., 'gap_sum': 0.}
    rows, start_offsets = [], []
    for pair in pairs:
        prefix = pair['prefix_ids']
        for key in ('chosen_ids', 'rejected_ids'):
            suffix = pair[key]
            rows.append({'input_ids': prefix+suffix, 'labels': [-100]*len(prefix)+suffix})
            start_offsets.append(len(prefix)+pair['difference_start'])
    batch = CausalCollator(pad_id)(rows)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    was_training = model.training
    model.eval()  # Keep dropout from deciding a minimal-pair comparison; keep gradients.
    try:
        logits = model(batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
        targets = batch['labels'][:, 1:]
        token_losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), targets,
                                       reduction='none', ignore_index=-100)
        full_counts = targets.ne(-100).sum(-1)
        positive = (token_losses.sum(-1)/full_counts.clamp_min(1))[::2].mean()
        masks = torch.zeros_like(targets, dtype=torch.bool)
        for i, start in enumerate(start_offsets):
            masks[i, start-1:start-1+32] = True
        masks &= targets.ne(-100)
        nll = (token_losses.mul(masks).sum(-1)/masks.sum(-1).clamp_min(1)).reshape(-1, 2)
        gap = nll[:, 1]-nll[:, 0]
        preference = F.softplus(beta*(margin-gap)).mean()
        return preference, positive, {'pairs': len(pairs), 'wins': float(gap.detach().gt(0).sum()),
                                     'gap_sum': float(gap.detach().sum())}
    finally:
        model.train(was_training)


def score_continuity_pairs(model, pairs, pad_id, batch_size=8):
    import torch
    totals = defaultdict(lambda: defaultdict(float))
    with torch.inference_mode():
        for kind in sorted({p['kind'] for p in pairs}):
            subset = [p for p in pairs if p['kind'] == kind]
            for start in range(0, len(subset), batch_size):
                batch = subset[start:start+batch_size]
                _, positive, stats = continuity_preference_loss(model, batch, pad_id)
                for key in ('all', kind):
                    for metric, value in stats.items():
                        totals[key][metric] += value
                    totals[key]['positive_nll_sum'] += float(positive)*len(batch)
    return {kind: {'pairs': int(s['pairs']), 'preference_accuracy': s['wins']/s['pairs'],
                   'mean_gap_nll': s['gap_sum']/s['pairs'], 'positive_nll': s['positive_nll_sum']/s['pairs']}
            for kind, s in totals.items()}


def reference_preference_loss(model, reference, pair, pad_id, beta=2.):
    """Prefer source over a model rollout relative to frozen starting weights.

    Length-normalized log ratios avoid rewarding a mere change in response
    length. This is a DPO-style objective; source preference is not a claim that
    every different continuation is semantically wrong.
    """
    import torch
    import torch.nn.functional as F
    prefix = pair['prefix_ids']
    rows = [{'input_ids': prefix+pair[key], 'labels': [-100]*len(prefix)+pair[key]}
            for key in ('chosen_ids', 'rejected_ids')]
    batch = {k: v.to(model.device) for k, v in CausalCollator(pad_id)(rows).items()}
    was_training = model.training
    model.eval()
    reference.eval()
    try:
        def gap(network):
            logits = network(batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
            sums = loss_sums(logits, batch['labels'])
            nll = sums[:, 0]/sums[:, 1].clamp_min(1)
            return nll[1]-nll[0]
        with torch.no_grad():
            anchor = gap(reference)
        advantage = gap(model)-anchor
        return F.softplus(-beta*advantage), float(advantage.detach())
    finally:
        model.train(was_training)


def online_continuity_loss(trainer, model, pairs):
    """Mine a fresh training-only rollout; never use generated prose as positive."""
    if not pairs:
        return next(model.parameters()).new_zeros(()), 0, 0.
    cfg, step = trainer.coherence_config, trainer.state.global_step
    pair = pairs[(step//cfg['online_every']) % len(pairs)]
    mode = 'sample' if (step//cfg['online_every']) % 2 else 'greedy'
    budget = min(cfg['online_tokens'], len(pair['chosen_ids']))
    prompt = {'input_ids': pair['prefix_ids'], 'prompt': ''}
    tokenizer = trainer.processing_class
    if tokenizer is None:
        from types import SimpleNamespace
        tokenizer = SimpleNamespace(eos_token_id=model.config.eos_token_id, pad_token_id=model.config.pad_token_id,
                                    decode=lambda ids, **kwargs: '')  # Mining only needs original token IDs.
    sample, = generate_samples(model, tokenizer, [prompt], budget,
                               seed=trainer.args.seed+step, modes=(mode,), include_token_ids=True)
    rejected = sample['token_ids']
    chosen = pair['chosen_ids'][:budget]
    if not rejected or chosen == rejected:
        return next(model.parameters()).new_zeros(()), 0, 0.
    mined = {**pair, 'chosen_ids': chosen, 'rejected_ids': rejected}
    loss, advantage = reference_preference_loss(model, trainer.continuity_reference, mined,
                                                model.config.pad_token_id, cfg['online_beta'])
    # Small auditable sample; avoid filling Colab storage with every rollout.
    if step % (cfg['online_every']*25) == 0:
        record = {'step': step, 'book_id': pair['book_id'], 'mode': mode,
                  'prefix_ids': pair['prefix_ids'], 'chosen_ids': chosen, 'rejected_ids': rejected,
                  'reference_relative_gap': advantage}
        with (Path(trainer.args.output_dir)/'online_negatives.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record)+'\n')
    return loss, 1, advantage


class CoherenceDataset:
    """Annotate original windows; no fabricated positive text or removed books."""
    def __init__(self, base, tokenizer, continuity_pairs=None):
        self.base, self.tokenizer = base, tokenizer
        self.block_size, self.random_offsets = base.block_size, base.random_offsets
        self.continuity = defaultdict(list)
        for pair in continuity_pairs or []:
            self.continuity[str(pair['book_id'])].append(pair)

    @property
    def epoch(self):
        return self.base.epoch

    def set_training_epoch(self, epoch):
        self.base.set_training_epoch(epoch)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = self.base[index]
        pool, dataset, window = 0, self.base, index
        if isinstance(self.base, WinterReplayDataset):
            slot, replay = divmod(index, 2)
            pool = int(not replay)
            if replay:
                dataset, window = self.base.replay, slot
            else:
                cycle, window = self.base.order[slot]
                dataset = self.base.cycles[cycle]
        book, _ = dataset.windows[window]
        key = int.from_bytes(hashlib.sha256(dataset.book_ids[book].encode()).digest()[:7], 'big')
        text = self.tokenizer.decode(row['input_ids'], skip_special_tokens=True)
        result = {**row, 'book_key': key, 'pool': pool, 'simplicity': passage_simplicity(text)}
        candidates = self.continuity.get(dataset.book_ids[book], [])
        if candidates:
            # Deterministic indexed rotation survives restored epochs/loader skips.
            result['continuity_pair'] = candidates[(index+self.epoch*104729) % len(candidates)]
        return result


def curriculum_weights(simplicity, pools, counts, progress, fraction):
    """Anneal to uniform; preserve each pool's total weighted targets per batch."""
    import torch
    strength = max(0., 1-progress/fraction) if fraction else 0.
    weights = 1+strength*(simplicity-1)
    for pool in torch.unique(pools):
        mask = pools == pool
        weights[mask] *= counts[mask].sum()/(weights[mask]*counts[mask]).sum().clamp_min(1)
    return weights


def context_pair_loss(model, inputs, labels, book_keys, margin, prefix=128, targets=64):
    """Rank true source context above different-book context for the SAME suffix."""
    import torch
    rows, suffixes = [], []
    active_positions = [torch.where(row != -100)[0] for row in labels]
    for i in range(len(inputs)):
        active = active_positions[i]
        if len(active) < targets or int(active[0]) < prefix:
            continue
        start = int(active[0])
        other = next((j for j in range(len(inputs)) if book_keys[j] != book_keys[i]
                      and len(active_positions[j]) and int(active_positions[j][0]) >= prefix), None)
        if other is None:
            continue
        other_start = int(active_positions[other][0])
        suffix = inputs[i, start:start+targets]
        if torch.equal(inputs[i, start-prefix:start], inputs[other, other_start-prefix:other_start]):
            continue
        rows += [torch.cat((inputs[i, start-prefix:start], suffix)),
                 torch.cat((inputs[other, other_start-prefix:other_start], suffix))]
        suffixes.append(suffix)
    if not rows:
        return inputs.new_zeros((), dtype=torch.float32), 0
    batch = torch.stack(rows)
    target = batch.clone()
    target[:, :prefix] = -100
    was_training = model.training
    model.eval()  # Paired differences must not be dropout noise; gradients remain on.
    try:
        logits = model(batch, use_cache=False).logits
        sums = loss_sums(logits, target)
        nll = (sums[:, 0]/sums[:, 1]).reshape(-1, 2)
        return (margin+nll[:, 0]-nll[:, 1]).clamp_min(0).mean(), len(suffixes)
    finally:
        model.train(was_training)


def repetition_negative_positions(tokens, width=8):
    """Only third-and-later occurrences of an eight-token phrase are negatives."""
    seen, positions = defaultdict(int), []
    for end in range(width-1, len(tokens)):
        gram = tuple(tokens[end-width+1:end+1])
        if seen[gram] >= 2:
            positions.append(end)
        seen[gram] += 1
    return positions


def repetition_unlikelihood(model, inputs, attention_mask, tokens=64):
    """Mine one greedy continuation; punish loops, always alongside original CE."""
    import torch
    length = min(128, int(attention_mask[0].sum())-1)
    prefix = inputs[:1, :max(1, length)]
    budget = min(tokens, model.config.n_positions-prefix.shape[1])
    if budget <= 0:
        return inputs.new_zeros((), dtype=torch.float32), 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            generated = model.generate(prefix, attention_mask=torch.ones_like(prefix), do_sample=False,
                max_new_tokens=budget, use_cache=True, pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id, suppress_tokens=[model.config.pad_token_id])
        new = generated[0, prefix.shape[1]:].tolist()
        bad = repetition_negative_positions(new)
        if not bad:
            return inputs.new_zeros((), dtype=torch.float32), 0
        positions = torch.tensor(bad, device=inputs.device)+prefix.shape[1]
        logits = model(generated, use_cache=False).logits[0, positions-1].float()
        probabilities = logits.softmax(-1).gather(1, generated[0, positions, None]).squeeze(1)
        return -torch.log1p(-probabilities.clamp(max=1-1e-5)).mean(), len(bad)
    finally:
        model.train(was_training)


def coherence_loss(trainer, model, inputs, num_items_in_batch=None, return_outputs=False):
    import torch
    cfg = trainer.coherence_config
    batch = {k: v for k, v in inputs.items() if k not in {'simplicity', 'book_key', 'pool', 'continuity_pairs'}}
    outputs = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
    sums = loss_sums(outputs.logits, batch['labels'])
    counts = sums[:, 1].detach()
    denominator = (counts.sum() if num_items_in_batch is None else
                   torch.as_tensor(num_items_in_batch, device=counts.device)).clamp_min(1)
    progress = trainer.state.global_step/max(1, trainer.state.max_steps)
    weights = curriculum_weights(inputs['simplicity'], inputs['pool'], counts, progress, cfg['curriculum_fraction'])
    loss = (sums[:, 0]*weights).sum()/denominator
    contrast, pairs = context_pair_loss(model, batch['input_ids'], batch['labels'], inputs['book_key'],
                                       cfg['context_margin']) if cfg['context_weight'] else (loss.new_zeros(()), 0)
    loss = loss+cfg['context_weight']*contrast*counts.sum()/denominator
    preference, positive, pair_stats = loss.new_zeros(()), loss.new_zeros(()), {'pairs': 0, 'wins': 0., 'gap_sum': 0.}
    if cfg.get('continuity_weight', 0):
        local_pairs = inputs.get('continuity_pairs', [])[:cfg['continuity_pairs_per_batch']]
        preference, positive, pair_stats = continuity_preference_loss(
            model, local_pairs, model.config.pad_token_id, cfg['continuity_beta'], cfg['continuity_margin'])
        loss = loss+(cfg['continuity_weight']*preference+cfg['continuity_positive_weight']*positive)*counts.sum()/denominator
    ul, negatives = loss.new_zeros(()), 0
    step = trainer.state.global_step
    # Exactly once per optimizer step, including after an optimizer-boundary resume.
    if cfg['unlikelihood_weight'] and step % cfg['unlikelihood_every'] == 0 and trainer._ul_step != step:
        trainer._ul_step = step
        ul, negatives = repetition_unlikelihood(model, batch['input_ids'], batch['attention_mask'], cfg['unlikelihood_tokens'])
        loss = loss+cfg['unlikelihood_weight']*ul
    online, online_pairs, advantage = loss.new_zeros(()), 0, 0.
    if (cfg.get('online_weight', 0) and inputs.get('continuity_pairs')
            and step % cfg['online_every'] == 0 and trainer._online_step != step):
        trainer._online_step = step
        online, online_pairs, advantage = online_continuity_loss(trainer, model, inputs.get('continuity_pairs', []))
        loss = loss+cfg['online_weight']*online  # Once per update, not once per microbatch.
    stats = trainer.coherence_stats
    stats['ce_sum'] += float(sums[:, 0].detach().sum())
    stats['tokens'] += float(counts.sum())
    stats['context_sum'] += float(contrast.detach())*pairs
    stats['pairs'] += pairs
    stats['ul_sum'] += float(ul.detach())*negatives
    stats['negatives'] += negatives
    stats['continuity_loss'] += float(preference.detach())*pair_stats['pairs']
    stats['continuity_positive_nll'] += float(positive.detach())*pair_stats['pairs']
    for key, value in pair_stats.items():
        stats['continuity_'+key] += value
    stats['online_loss'] += float(online.detach())*online_pairs
    stats['online_pairs'] += online_pairs
    stats['online_advantage'] += advantage
    return (loss, outputs) if return_outputs else loss


def training_trainer(*args, **kwargs):
    """Shuffle from the restored training epoch, including a resumed skip loader."""
    import torch
    from torch.utils.data import Sampler
    from transformers import Trainer
    coherence_config = kwargs.pop('coherence_config', None)

    class EpochSampler(Sampler):
        def __init__(self, dataset, seed):
            self.dataset, self.seed = dataset, seed

        def __iter__(self):
            generator = torch.Generator().manual_seed(self.seed+self.dataset.epoch)
            return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

        def __len__(self):
            return len(self.dataset)

    class WindowTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.coherence_config = coherence_config
            self.coherence_stats, self._ul_step = defaultdict(float), -1
            self._online_step = -1
            self.continuity_reference = None
            if coherence_config and coherence_config.get('online_weight', 0):
                import copy
                # Created before Trainer loads resumed optimizer/student weights:
                # the caller supplies the same frozen starting model on every resume.
                self.continuity_reference = copy.deepcopy(self.model).eval().requires_grad_(False)
            if coherence_config:
                # Our CE and per-microbatch margin use the accumulation-wide token denominator.
                self.model_accepts_loss_kwargs = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if self.coherence_config and model.training:
                return coherence_loss(self, model, inputs, num_items_in_batch, return_outputs)
            clean = {k: v for k, v in inputs.items() if k not in {'simplicity', 'book_key', 'pool', 'continuity_pairs'}}
            return super().compute_loss(model, clean, return_outputs, num_items_in_batch)

        def log(self, logs, start_time=None):
            if self.coherence_config and self.coherence_stats['tokens'] and ('loss' in logs or 'train_loss' in logs):
                s = self.coherence_stats
                logs = {**logs, 'ce_nll': s['ce_sum']/max(1, s['tokens']),
                        'context_margin_loss': s['context_sum']/max(1, s['pairs']),
                        'context_pairs': s['pairs'], 'unlikelihood_loss': s['ul_sum']/max(1, s['negatives']),
                        'unlikelihood_negative_tokens': s['negatives']}
                if self.coherence_config.get('continuity_weight', 0):
                    logs.update(continuity_pairs=s['continuity_pairs'],
                        continuity_loss=s['continuity_loss']/max(1, s['continuity_pairs']),
                        continuity_positive_nll=s['continuity_positive_nll']/max(1, s['continuity_pairs']),
                        continuity_accuracy=s['continuity_wins']/max(1, s['continuity_pairs']),
                        continuity_gap=s['continuity_gap_sum']/max(1, s['continuity_pairs']))
                if self.coherence_config.get('online_weight', 0):
                    logs.update(online_preference_pairs=s['online_pairs'],
                        online_preference_loss=s['online_loss']/max(1, s['online_pairs']),
                        online_reference_relative_gap=s['online_advantage']/max(1, s['online_pairs']))
                self.coherence_stats = defaultdict(float)
            return super().log(logs, start_time)

        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            seed = self.args.seed if self.args.data_seed is None else self.args.data_seed
            # Unlike RandomSampler, this will not be replaced by Accelerate's
            # SeedableRandomSampler, whose epoch the skipped loader can reset.
            return EpochSampler(dataset, seed)

    return WindowTrainer(*args, **kwargs)


class CausalCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        import torch
        width = max(len(x["input_ids"]) for x in examples)
        ids = torch.full((len(examples), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, example in enumerate(examples):
            n = len(example["input_ids"])
            ids[i, :n] = torch.tensor(example["input_ids"])
            supplied = example.get("labels", example["input_ids"])
            if len(supplied) != n:
                raise ValueError("Labels must align with input IDs before causal shifting")
            labels[i, :n] = torch.tensor(supplied)
            mask[i, :n] = 1
        # Position zero is context, never a next-token target. Mask it explicitly
        # so Trainer's accumulation denominator counts the shifted targets too.
        labels[:, 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def loss_sums(logits, labels):
    """Reduce logits before storing evaluation outputs; retain token-weighted NLL."""
    import torch
    from torch.nn import functional as F
    if isinstance(logits, tuple):
        logits = logits[0]
    targets = labels[:, 1:]
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), targets,
                             reduction="none", ignore_index=-100)
    return torch.stack((losses.sum(dim=1), (targets != -100).sum(dim=1)), dim=1)


def token_metrics(prediction):
    values = prediction.predictions
    nll = float(values[:, 0].sum(dtype="float64") / values[:, 1].sum(dtype="float64"))
    return {"nll": nll, "perplexity": math.exp(min(nll, 50)), "scored_tokens": int(values[:, 1].sum())}


def contrastive_generate(model, inputs, max_new_tokens, pad_id, eos_id, top_k=4, alpha=.6):
    """Single unpadded GPT-2 sequence; caller supplies eval/inference mode.

    Rank top-k candidates by (1-alpha)*probability - alpha*max_cosine_to_context.
    Cache branches are copied, then only the chosen branch is retained. Kept
    inline for standalone Colab use without downloading custom generation code.
    """
    import torch
    from torch.nn import functional as F
    from transformers import DynamicCache
    if inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.shape[1] == 0:
        raise ValueError("Contrastive search requires one nonempty, unpadded prompt")
    if not 0 <= alpha < 1 or not 1 <= top_k < model.config.vocab_size:
        raise ValueError("Contrastive alpha must be in [0,1); top-k must be in [1,vocab_size)")
    room = model.config.n_positions - inputs.shape[1]
    if room <= 0 or max_new_tokens <= 0:
        raise ValueError("Shorten the prompt and use a positive generation length")
    output = model(inputs, attention_mask=torch.ones_like(inputs), use_cache=True,
                   output_hidden_states=True, return_dict=True)
    context = F.normalize(output.hidden_states[-1][0].float(), dim=-1)
    cache = output.past_key_values.to_legacy_cache()
    logits = output.logits[:, -1]
    sequence = inputs
    for _ in range(min(max_new_tokens, room)):
        logits = logits.float().clone()
        logits[:, pad_id] = -float("inf")
        probabilities, candidates = logits.softmax(dim=-1).topk(top_k, dim=-1)
        # DynamicCache is mutable. Separate candidate branches must never share
        # the selected prefix cache's writable storage.
        branches = DynamicCache.from_legacy_cache(tuple(
            (key.repeat_interleave(top_k, dim=0), value.repeat_interleave(top_k, dim=0))
            for key, value in cache))
        candidate_output = model(candidates.reshape(top_k, 1), past_key_values=branches,
            attention_mask=torch.ones((top_k, sequence.shape[1]+1), dtype=torch.long, device=inputs.device),
            use_cache=True, output_hidden_states=True, return_dict=True)
        hidden = F.normalize(candidate_output.hidden_states[-1][:, -1].float(), dim=-1)
        penalty = (hidden @ context.T).max(dim=-1).values
        selected = int(((1-alpha)*probabilities[0] - alpha*penalty).argmax())
        token = candidates[:, selected:selected+1]
        sequence = torch.cat((sequence, token), dim=1)
        if int(token.item()) == eos_id:
            break
        context = torch.cat((context, hidden[selected:selected+1]), dim=0)
        logits = candidate_output.logits[selected:selected+1, -1]
        cache = tuple((key[selected:selected+1].contiguous(), value[selected:selected+1].contiguous())
                      for key, value in candidate_output.past_key_values.to_legacy_cache())
    return sequence


class CoherenceCollator(CausalCollator):
    def __call__(self, examples):
        import torch
        batch = super().__call__(examples)
        batch['simplicity'] = torch.tensor([r.get('simplicity', 1.) for r in examples])
        batch['book_key'] = torch.tensor([r.get('book_key', 0) for r in examples])
        batch['pool'] = torch.tensor([r.get('pool', 0) for r in examples])
        batch['continuity_pairs'] = [r['continuity_pair'] for r in examples if 'continuity_pair' in r]
        return batch


def rank_continuations(model, tokenizer, prefix, candidates):
    """Same-model heuristics, not an independent coherence judge. No external LM."""
    import torch
    if not prefix or not len(candidates):
        raise ValueError('Reranking requires a nonempty prefix and candidates')
    rows = []
    for index, tokens in enumerate(candidates):
        ids = list(tokens)
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)+1]
        ids = [t for t in ids if t != tokenizer.pad_token_id]
        if not ids:
            rows.append({'candidate': index, 'score': -1e9, 'token_ids': []})
            continue
        scores = []
        for context in (prefix, prefix[-min(16, len(prefix)):]):
            inputs = torch.tensor([context+ids], device=model.device)
            positions = torch.arange(len(prefix)-len(context), len(prefix)+len(ids), device=model.device)[None]
            labels = inputs.clone()
            labels[:, :len(context)] = -100
            logits = model(inputs, position_ids=positions, use_cache=False).logits
            sums = loss_sums(logits, labels)[0]
            scores.append(float((sums[0]/sums[1]).detach()))
        content = [t for t in ids if t != tokenizer.eos_token_id]
        grams = [tuple(content[i:i+4]) for i in range(max(0, len(content)-3))]
        repeated = (len(grams)-len(set(grams)))/max(1, len(grams))
        prompt_grams = {tuple(prefix[i:i+8]) for i in range(max(0, len(prefix)-7))}
        echo = sum(tuple(content[i:i+8]) in prompt_grams for i in range(max(0, len(content)-7)))/max(1, len(content)-7)
        gain = scores[1]-scores[0]
        score = -scores[0]+.3*max(-2., min(2., gain))-4*repeated-2*echo
        # Avoid selecting an immediate/near-immediate EOS just for high likelihood.
        # This is a ranking penalty, never a forced minimum generation length.
        score -= 2*max(0., 1-len(content)/min(24, max(map(len, candidates))))
        if not content:
            score = -1e9
        rows.append({'candidate': index, 'score': score, 'conditional_nll': scores[0],
                     'short_context_nll': scores[1], 'context_gain_nll': gain,
                     'repeated_4gram_fraction': repeated, 'prompt_echo_fraction': echo,
                     'token_ids': ids, 'continuation': tokenizer.decode(ids, skip_special_tokens=True)})
    best = max(range(len(rows)), key=lambda i: rows[i]['score'])
    return rows[best]['token_ids'], {'selected_candidate': best, 'candidates': rows,
        'interpretation': 'Fluency, full-vs-short-context gain and repetition heuristics; not a coherence score.'}


def rerank_generate(model, tokenizer, inputs, budget, candidates=8):
    output = model.generate(inputs, attention_mask=inputs.new_ones(inputs.shape), do_sample=True,
        num_return_sequences=candidates, max_new_tokens=budget, temperature=.8, top_p=.92, top_k=40,
        repetition_penalty=1.08, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, suppress_tokens=[tokenizer.pad_token_id], use_cache=True)
    return rank_continuations(model, tokenizer, inputs[0].tolist(),
                              [row[inputs.shape[1]:].tolist() for row in output])


def generate_samples(model, tokenizer, prompts, max_new_tokens=96, seed=1234,
                     modes=("greedy", "sample"), include_token_ids=False):
    import torch
    device = next(model.parameters()).device
    was_training = model.training
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    # Sampling must not change the random stream used by training dropout.
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    results = []
    model.eval()
    try:
        with torch.inference_mode(), torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            for prompt in prompts:
                if isinstance(prompt, dict):
                    # Corpus probes use original IDs so a window beginning or
                    # ending inside a UTF-8/BPE span is not re-tokenized.
                    ids, prompt = list(prompt["input_ids"]), prompt["prompt"]
                else:
                    ids = tokenizer.encode(prompt, add_special_tokens=False) or [tokenizer.eos_token_id]
                room = model.config.n_positions - len(ids)
                if room <= 0:
                    raise ValueError("Prompt fills the context window; shorten it")
                inputs = torch.tensor([ids], dtype=torch.long, device=device)
                for mode in modes:
                    if mode not in {"greedy", "sample", "blocked_sample", "contrastive", "reranked"}:
                        raise ValueError(f"Unknown decoding mode: {mode}")
                    kwargs = dict(max_new_tokens=min(max_new_tokens, room),
                                  do_sample=mode in {"sample", "blocked_sample"}, use_cache=True,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                  suppress_tokens=[tokenizer.pad_token_id])
                    if mode in {"sample", "blocked_sample"}:
                        kwargs.update(temperature=.8, top_p=.92, top_k=40, repetition_penalty=1.08)
                    if mode == "blocked_sample":
                        kwargs["no_repeat_ngram_size"] = 4
                    ranking = None
                    if mode == "reranked":
                        tokens, ranking = rerank_generate(model, tokenizer, inputs, min(max_new_tokens, room),
                            getattr(model.config, 'coherence_decoding', {}).get('candidates', 8))
                    elif mode == "contrastive":
                        output = contrastive_generate(model, inputs, min(max_new_tokens, room),
                                                      tokenizer.pad_token_id, tokenizer.eos_token_id)
                    else:
                        output = model.generate(inputs, attention_mask=torch.ones_like(inputs), **kwargs)
                    if mode != "reranked":
                        tokens = output[0, len(ids):].tolist()
                    row = {"prompt": prompt, "mode": mode,
                           "continuation": tokenizer.decode(tokens, skip_special_tokens=True)}
                    if include_token_ids:
                        row["token_ids"] = tokens
                    if ranking is not None:
                        row['ranking'] = ranking
                    results.append(row)
    finally:
        model.train(was_training)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
    return results


def print_sample(sample, label=None):
    tag = f"{label} | {sample['mode']}" if label else sample['mode']
    print(f"\n[{tag}]\nPROMPT:\n{sample['prompt']}\n\nGENERATED:\n{sample['continuation']}\n", flush=True)


def dataset_signature(data):
    digest = hashlib.sha256()
    for name in ("report.json", "train.jsonl", "validation.jsonl", "test.jsonl", "tokenizer/tokenizer.json"):
        with (Path(data) / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_continuation(source, tokenizer, signature):
    """Load this project's own weights, checking the original data and token IDs."""
    from transformers import AutoTokenizer, GPT2LMHeadModel
    source = Path(source).resolve()
    records = [source / "provenance.json", source / "training_summary.json", source.parent / "run_config.json"]
    record = next((path for path in records if path.is_file()), None)
    if record is None:
        raise ValueError("Continuation requires a project best/ or checkpoint directory with dataset provenance")
    if json.loads(record.read_text()).get("dataset_sha256") != signature:
        raise ValueError("Continuation dataset differs from the source run; keep its prepared data/tokenizer")
    saved_tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if (saved_tokenizer.backend_tokenizer.to_str() != tokenizer.backend_tokenizer.to_str()
            or saved_tokenizer.special_tokens_map != tokenizer.special_tokens_map):
        raise ValueError("Continuation tokenizer differs from the prepared tokenizer")
    model = GPT2LMHeadModel.from_pretrained(source, local_files_only=True)
    if (model.config.vocab_size != len(tokenizer)
            or model.config.eos_token_id != tokenizer.eos_token_id
            or model.config.pad_token_id != tokenizer.pad_token_id):
        raise ValueError("Continuation model and tokenizer IDs disagree")
    return model


def select_continuation_result(model, initial, initial_nll, trained_nll, trained_checkpoint):
    """The continuation must beat its starting model on the same validation data."""
    from transformers import GPT2LMHeadModel
    retained_initial = trained_nll is None or initial_nll <= trained_nll
    if retained_initial:
        original = GPT2LMHeadModel.from_pretrained(initial, local_files_only=True)
        model.load_state_dict(original.state_dict())
        return str(initial), initial_nll, True
    return trained_checkpoint, trained_nll, False


def set_continuation_dropout(model, rate):
    """Change both saved config and already-instantiated GPT-2 dropout modules."""
    import torch
    if not 0 <= rate < 1:
        raise ValueError("Continuation dropout must be in [0,1)")
    for field in ("resid_pdrop", "embd_pdrop", "attn_pdrop"):
        setattr(model.config, field, rate)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = rate


def grow_gpt2_depth(source, layers):
    """Insert identity residual blocks while retaining a standard GPT-2 model."""
    import copy

    import torch
    from transformers import GPT2LMHeadModel
    previous = source.config.n_layer
    if layers <= previous:
        raise ValueError(f"New depth must exceed the source model's {previous} layers")
    if (source.config.add_cross_attention or source.config.scale_attn_by_inverse_layer_idx
            or not source.config.tie_word_embeddings
            or source.lm_head.weight.data_ptr() != source.transformer.wte.weight.data_ptr()):
        raise ValueError("Depth growth requires plain GPT-2 with tied embeddings and no layer-index attention scaling")
    config = copy.deepcopy(source.config)
    config.n_layer = layers
    grown = GPT2LMHeadModel(config).to(device=source.device, dtype=source.dtype)
    grown.generation_config = copy.deepcopy(source.generation_config)
    mapping = [index*layers//previous for index in range(previous)]
    inserted = [index for index in range(layers) if index not in mapping]
    grown.transformer.wte.load_state_dict(source.transformer.wte.state_dict())
    grown.transformer.wpe.load_state_dict(source.transformer.wpe.state_dict())
    grown.transformer.ln_f.load_state_dict(source.transformer.ln_f.state_dict())
    for old, new in enumerate(mapping):
        grown.transformer.h[new].load_state_dict(source.transformer.h[old].state_dict())
    with torch.no_grad():
        for index in inserted:
            # Each pre-LN GPT-2 block adds attention and MLP outputs to its
            # input. Zero output projections make these two residuals zero.
            # The other new weights stay random so the block can learn.
            for projection in (grown.transformer.h[index].attn.c_proj, grown.transformer.h[index].mlp.c_proj):
                projection.weight.zero_()
                projection.bias.zero_()
    grown.tie_weights()
    grown.train(source.training)
    return grown, {"method": "interleaved_identity_residual_blocks", "source_layers": previous,
                   "target_layers": layers, "old_layer_positions": mapping, "inserted_layers": inserted,
                   "source_parameters": source.num_parameters(), "target_parameters": grown.num_parameters()}


def verify_depth_growth(source, grown, tokenizer):
    """Reject an expansion that changes logits on the fixed opening probes."""
    import torch
    source_mode, grown_mode = source.training, grown.training
    source.eval()
    grown.eval()
    maximum = 0.
    try:
        with torch.inference_mode():
            for prompt in PROMPTS:
                ids = tokenizer.encode(prompt, add_special_tokens=False)[:source.config.n_positions]
                inputs = torch.tensor([ids], device=source.device)
                expected = source(inputs, use_cache=False).logits
                actual = grown(inputs, use_cache=False).logits
                maximum = max(maximum, float((actual-expected).abs().max()))
                if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
                    raise ValueError("Depth growth changed initial logits; refusing to start training")
    finally:
        source.train(source_mode)
        grown.train(grown_mode)
    return {"prompts_checked": len(PROMPTS), "max_absolute_logit_difference": maximum,
            "dropout_disabled": True, "rtol": 1e-5, "atol": 1e-6}


def latest_complete_checkpoint(directory, require_scaler=False):
    """Ignore a checkpoint directory interrupted before its state files finished."""
    required = ("model.safetensors", "config.json", "optimizer.pt", "scheduler.pt",
                "rng_state.pth", "trainer_state.json")
    if require_scaler:
        required += ("scaler.pt",)
    candidates = [path for path in Path(directory).glob("checkpoint-*")
                  if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()]
    for path in sorted(candidates, key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        try:
            if (all((path / name).is_file() and (path / name).stat().st_size > 0 for name in required)
                    and json.loads((path / "trainer_state.json").read_text())["global_step"]
                    == int(path.name.split("-")[-1])):
                return str(path)
        except (OSError, ValueError, KeyError):
            continue
    return None


def training(args):
    import torch
    from transformers import (
        AutoTokenizer,
        EarlyStoppingCallback,
        GPT2Config,
        GPT2LMHeadModel,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint
    fixed_budget = getattr(args, "fixed_budget", False)
    scene_recipe = getattr(args, "scene_recipe", None)
    winter_recipe = getattr(args, "winter_recipe", None)
    coherence_config = (winter_recipe or {}).get('coherence')
    attention_dropout = getattr(args, "attention_dropout", .1)
    if (args.resume or args.continue_from) and args.rebuild:
        raise ValueError("Do not rebuild a dataset while resuming or continuing an existing model")
    report = prepare(args)
    if args.smoke_test and not fixed_budget:
        args.layers, args.width, args.heads, args.block_size = 2, 64, 2, 64
        args.batch_size, args.accumulation, args.max_steps = 2, 1, 3
        args.eval_steps, args.sample_steps, args.sample_tokens = 1, 1, 12
        args.train_limit, args.eval_limit, args.cpu = 16, 8, True
    precision = resolve_precision(getattr(args, "precision", "auto"), args.cpu)
    model_seed = getattr(args, "model_seed", None)
    model_seed = args.seed if model_seed is None else model_seed
    set_seed(model_seed)
    data, out = Path(args.data_dir), Path(args.output_dir)
    resume = None
    if args.resume:
        if args.resume == "latest":
            resume = (latest_complete_checkpoint(out, require_scaler=precision == "fp16")
                      if fixed_budget else get_last_checkpoint(str(out)))
        else:
            resume = str(Path(args.resume).resolve())
        if not resume:
            if not fixed_budget or not (out / "run_config.json").exists():
                raise ValueError("No checkpoint found to resume")
            print("No saved optimizer checkpoint yet; restarting this fresh-model branch from its seed.", flush=True)
        if resume and Path(resume).resolve().parent != out.resolve():
            raise ValueError("Resume checkpoint must belong to this --output-dir")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; use --resume or a new --output-dir")
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    signature = dataset_signature(data)
    continuation = None
    initial = out / "initial"
    if args.continue_from:
        source = Path(args.continue_from).resolve()
        if out.resolve() == source or out.resolve() in source.parents:
            raise ValueError("Continuation needs a separate output directory from its source run")
        model = load_continuation(source, tokenizer, signature)
        continuation = {"source": str(source), "dataset_sha256": signature}
        if args.grow_layers is not None:
            grown, growth = grow_gpt2_depth(model, args.grow_layers)
            growth["verification"] = verify_depth_growth(model, grown, tokenizer)
            continuation["depth_growth"] = growth
            model = grown
            print(f"Depth growth: {growth['source_layers']} -> {growth['target_layers']} layers; "
                  f"{growth['target_parameters']:,} parameters. "
                  f"Initial logit check: {growth['verification']}", flush=True)
        if args.continuation_dropout is not None:
            continuation["original_dropout"] = {field: getattr(model.config, field)
                for field in ("resid_pdrop", "embd_pdrop", "attn_pdrop")}
            set_continuation_dropout(model, args.continuation_dropout)
            continuation["dropout_override"] = args.continuation_dropout
    elif resume and (out / "continuation.json").exists():
        continuation = json.loads((out / "continuation.json").read_text())
        model = load_continuation(initial, tokenizer, signature)
    if continuation is not None:
        # Inherit the source architecture unless explicitly grown above. Ordinary
        # CLI size settings apply only to new randomly initialized models.
        args.layers, args.width, args.heads = model.config.n_layer, model.config.n_embd, model.config.n_head
        args.block_size, args.dropout = model.config.n_positions, model.config.resid_pdrop
        attention_dropout = model.config.attn_pdrop
        model.config.use_cache = False
    tokenizer.model_max_length = args.block_size
    settings = {"dataset_sha256": signature, "layers": args.layers, "width": args.width,
                "heads": args.heads, "block_size": args.block_size, "dropout": args.dropout,
                "batch_size": args.batch_size, "accumulation": args.accumulation,
                "learning_rate": args.learning_rate, "epochs": args.epochs, "max_steps": args.max_steps,
                "seed": args.seed, "model_seed": model_seed,
                "train_limit": args.train_limit, "eval_limit": args.eval_limit,
                "eval_steps": args.eval_steps, "patience": args.patience, "smoke_test": args.smoke_test,
                "warmup_ratio": args.warmup_ratio, "skip_test": args.skip_test,
                "random_window_offsets": args.random_window_offsets,
                "attention_dropout": attention_dropout, "precision": precision,
                "fixed_budget": fixed_budget,
                "benchmark_sha256": getattr(args, "benchmark_sha256", None),
                "benchmark_steps": args.benchmark_steps if fixed_budget else None}
    if continuation is not None:
        settings["continue_from"] = continuation["source"]
    settings["grow_layers"] = (continuation or {}).get("depth_growth", {}).get("target_layers")
    if scene_recipe is not None:
        settings["scene_recipe"] = scene_recipe
    if winter_recipe is not None:
        settings["winter_recipe"] = winter_recipe
    if args.resume:
        prior = json.loads((out / "run_config.json").read_text())
        prior.setdefault("warmup_ratio", .05)  # Compatibility with the original script.
        prior.setdefault("skip_test", False)
        prior.setdefault("random_window_offsets", False)
        prior.setdefault("grow_layers", None)
        prior.setdefault("model_seed", prior["seed"])
        # Historical runs did not record these fields. New runs check them.
        for key, value in {"attention_dropout": attention_dropout if continuation else .1, "precision": precision,
                           "fixed_budget": False, "benchmark_sha256": None,
                           "benchmark_steps": None}.items():
            prior.setdefault(key, value)
        if prior != settings:
            raise ValueError("Run settings or dataset differ from saved run_config.json; use a new run")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", settings)
    if continuation is None:
        config = GPT2Config(vocab_size=len(tokenizer), n_positions=args.block_size, n_ctx=args.block_size,
                            n_embd=args.width, n_layer=args.layers, n_head=args.heads, n_inner=4*args.width,
                            activation_function="gelu_new", resid_pdrop=args.dropout,
                            embd_pdrop=args.dropout, attn_pdrop=attention_dropout, initializer_range=.02,
                            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id, tie_word_embeddings=True, use_cache=False)
        # Initial runs initialize ALL weights randomly. Continuations load only
        # the user's existing project model, never external pretrained GPT-2.
        model = GPT2LMHeadModel(config)
    parameters = model.num_parameters()
    train_set = BookWindows(data / "train.jsonl", args.block_size, args.train_limit,
                            random_offsets=args.random_window_offsets, seed=args.seed)
    review_training_data = train_set
    if scene_recipe and scene_recipe["treatment"]:
        train_set = SceneReplayDataset(train_set, args.scene_records, tokenizer.eos_token_id, args.seed)
    if winter_recipe:
        review_block = 512 if coherence_config else args.block_size
        review_training_data = BookWindows(data/'train.jsonl', review_block, args.train_limit)
        supplementary = BookWindows(args.winter_train_path, review_block, args.train_limit,
                                    random_offsets=True, seed=args.seed)
        # Check copying against BOTH pools, including for baseline/control prose.
        review_training_data = combined_book_windows(review_training_data, supplementary)
        # Both arms use the same context policy; review NLL retains the old windows.
        lookback = winter_recipe.get("lookback", 0)
        train_set = ContextBookWindows(data/"train.jsonl", args.block_size, args.train_limit,
                                       random_offsets=True, seed=args.seed, lookback=lookback)
        supplementary = ContextBookWindows(args.winter_train_path, args.block_size, args.train_limit,
                                           random_offsets=True, seed=args.seed, lookback=lookback)
        if winter_recipe["treatment"]:
            train_set = WinterReplayDataset(train_set, supplementary, args.seed)
    if coherence_config:
        train_set = CoherenceDataset(train_set, tokenizer, getattr(args, 'continuity_pairs', None))
    val_set = BookWindows(data / "validation.jsonl", 512 if coherence_config else args.block_size, args.eval_limit)
    effective_batch = args.batch_size * args.accumulation
    origin = "Continued" if continuation is not None else "Random"
    print(f"{origin} GPT-2: {parameters:,} parameters; {len(train_set):,} training examples; "
          f"effective batch {effective_batch} (at most {effective_batch*(args.block_size-1):,} targets/update).", flush=True)
    unique_targets = report['splits']['train']['target_tokens']
    if winter_recipe and winter_recipe["treatment"]:
        unique_targets += args.winter_manifest["target_tokens"]
    print(f"Unique training target tokens/parameter: {unique_targets/parameters:.2f}. "
          "Repeated epochs do not add new language evidence.", flush=True)
    use_cuda = torch.cuda.is_available() and not args.cpu
    bf16 = precision == "bf16"

    class SampleCallback(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            if state.global_step % args.sample_steps == 0 and state.is_world_process_zero:
                modes = ('greedy', 'sample', 'reranked') if coherence_config else ('greedy', 'sample')
                samples = generate_samples(kwargs["model"], tokenizer, PROMPTS, args.sample_tokens, modes=modes)
                with (out / "samples.jsonl").open("a", encoding="utf-8") as handle:
                    for sample in samples:
                        handle.write(json.dumps({"step": state.global_step, **sample}, ensure_ascii=False) + "\n")
                        print_sample(sample, f'step {state.global_step}')

    training_args = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation, learning_rate=args.learning_rate,
        weight_decay=.1, adam_beta1=.9, adam_beta2=.95, adam_epsilon=1e-8,
        max_grad_norm=1., warmup_ratio=args.warmup_ratio, lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": .1}, optim="adamw_torch",
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        logging_steps=min(25, args.eval_steps), save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="nll", greater_is_better=False,
        fp16=precision == "fp16", bf16=bf16, use_cpu=args.cpu,
        dataloader_num_workers=0, dataloader_pin_memory=use_cuda,
        seed=model_seed, data_seed=args.seed, report_to="none", save_safetensors=True,
        restore_callback_states_from_checkpoint=True,
        remove_unused_columns=not bool(coherence_config),
    )
    callbacks = [window_epoch_callback(train_set), SampleCallback()]
    capacity_hook = None
    if fixed_budget:
        capacity_hook = capacity_callback(args, tokenizer, review_training_data)
        callbacks.append(capacity_hook)
    else:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience,
                                              early_stopping_threshold=.002))
    trainer = training_trainer(
        model=model, args=training_args, train_dataset=train_set, eval_dataset=val_set,
        data_collator=(CoherenceCollator if coherence_config else CausalCollator)(tokenizer.pad_token_id),
        processing_class=tokenizer, coherence_config=coherence_config,
        preprocess_logits_for_metrics=loss_sums, compute_metrics=token_metrics,
        callbacks=callbacks)
    planned_steps = args.max_steps if args.max_steps > 0 else math.ceil(
        math.ceil(len(train_set)/effective_batch)*args.epochs)
    if planned_steps < args.eval_steps:
        raise ValueError("Training would finish before its first evaluation; lower --eval-steps")
    completed_resume = resume and json.loads((Path(resume) / "trainer_state.json").read_text())["global_step"] >= planned_steps
    if completed_resume and not fixed_budget:
        raise ValueError("This run already reached its step budget; use generate/evaluate or start a new run")
    if continuation is not None and not resume:
        # Snapshot the starting weights before creating a fresh optimizer. The
        # source remains untouched and resumption no longer needs its directory.
        model.save_pretrained(initial, safe_serialization=True)
        tokenizer.save_pretrained(initial)
        write_json(initial / "provenance.json", {"dataset_sha256": signature})
        baseline = trainer.evaluate()
        continuation["initial_validation_nll"] = baseline["eval_nll"]
        write_json(out / "continuation.json", continuation)
        write_json(out / "initial_samples.json", generate_samples(model, tokenizer, PROMPTS, args.sample_tokens))
        if scene_recipe is not None or winter_recipe is not None:
            run_capacity_review(model, tokenizer, args.capacity_benchmark, review_training_data,
                                initial, 0, baseline["eval_nll"])
        print(f"Continuation baseline NLL: {baseline['eval_nll']:.6f}. "
              "Fresh optimizer/schedule; update numbering restarts at zero.", flush=True)
    if completed_resume:
        # A Colab session can end after the final checkpoint but before best/
        # and its review are written. Finalize it without one extra update.
        from transformers.trainer_callback import TrainerState
        trainer.state = TrainerState.load_from_json(str(Path(resume) / "trainer_state.json"))
        if trainer.state.global_step != planned_steps:
            raise ValueError("Saved step exceeds the planned capacity budget")
        model.load_state_dict(GPT2LMHeadModel.from_pretrained(resume, local_files_only=True).state_dict())
        capacity_hook.record(model, trainer.state, force=True)
        model.load_state_dict(GPT2LMHeadModel.from_pretrained(
            trainer.state.best_model_checkpoint, local_files_only=True).state_dict())
    else:
        trainer.train(resume_from_checkpoint=resume)
    # Trainer has restored the best VALIDATION checkpoint, not necessarily the last.
    if continuation is not None:
        continuation["best_new_validation_nll"] = trainer.state.best_metric
        selected, score, retained = select_continuation_result(
            model, initial, continuation["initial_validation_nll"],
            trainer.state.best_metric, trainer.state.best_model_checkpoint)
        trainer.state.best_model_checkpoint, trainer.state.best_metric = selected, score
        continuation["retained_initial_model"] = retained
        write_json(out / "continuation.json", continuation)
        print(f"Continuation selection: {'starting model retained' if retained else 'improved checkpoint selected'}; "
              f"validation NLL {score:.6f}.", flush=True)
    model.config.use_cache = True
    best = out / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(best)
    trainer.save_state()
    trainer.remove_callback(EarlyStoppingCallback)
    metrics = {}
    if not args.skip_test:
        test_set = BookWindows(data / "test.jsonl", 512 if coherence_config else args.block_size, args.eval_limit)
        metrics = trainer.evaluate(test_set, metric_key_prefix="test")
    summary = {"parameters": parameters, "best_checkpoint": trainer.state.best_model_checkpoint,
               "best_validation_nll": trainer.state.best_metric, "global_step": trainer.state.global_step,
               "dataset_sha256": signature, "smoke_test": args.smoke_test,
               "limited_data": bool(args.train_limit or args.eval_limit), "test": metrics,
               "corpus": report, "run": settings, "continuation": continuation,
               "runtime": {"torch": str(torch.__version__), "device": str(trainer.args.device),
                           "precision": precision,
                           "gpu": torch.cuda.get_device_name() if use_cuda else None}}
    write_json(best / "training_summary.json", summary)
    if winter_recipe:
        write_json(best / "supplementary_sources.json", args.winter_manifest)
    if not args.skip_test:
        write_json(out / "test_metrics.json", metrics)
    final_samples = generate_samples(model, tokenizer, PROMPTS, args.sample_tokens)
    write_json(best / "samples.json", final_samples)
    if fixed_budget:
        summary["optimizer_skipped_steps"] = capacity_hook.skipped_count(trainer.state.global_step)
        summary["applied_optimizer_updates"] = trainer.state.global_step-summary["optimizer_skipped_steps"]
        summary["training_exposure"] = training_exposure(train_set, effective_batch,
                                                         trainer.state.global_step, args.seed)
        summary["best_validation_step"] = (0 if continuation and continuation["retained_initial_model"] else
                                          int(Path(trainer.state.best_model_checkpoint).name.split("-")[-1]))
        write_json(best / "training_summary.json", summary)
        run_capacity_review(model, tokenizer, args.capacity_benchmark, review_training_data,
                            best, summary["best_validation_step"], summary["best_validation_nll"])
    (best / "README.md").write_text(model_card(summary), encoding="utf-8")
    print(f"Best model and tokenizer: {best}", flush=True)
    print("Test evaluation skipped for model development." if args.skip_test else
          f"Held-out test: {json.dumps(metrics, indent=2)}", flush=True)
    return summary


def compare_dropout(args):
    """A paired, bounded experiment; reports loss and samples, not a coherence score."""
    import gc

    import torch
    if not args.continue_from or args.resume or args.rebuild:
        raise ValueError("compare-dropout requires --continue-from and a new output directory; no --resume/--rebuild")
    if args.max_steps <= 0:
        raise ValueError("Set an explicit --max-steps budget for each comparison branch (for example, 500)")
    if args.continuation_dropout is None:
        raise ValueError("Set --continuation-dropout for the treatment branch (for example, 0.05)")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new --output-dir for the paired comparison")
    results = {}
    for name, dropout in (("control", None), ("lower-dropout", args.continuation_dropout)):
        branch = argparse.Namespace(**vars(args))
        branch.command, branch.output_dir = "train", str(out / name)
        branch.continuation_dropout, branch.skip_test = dropout, True
        # Keep both branches on the same update budget, including if progress is flat.
        branch.patience = max(args.patience, math.ceil(args.max_steps / args.eval_steps) + 1)
        summary = training(branch)
        results[name] = {"dropout": dropout if dropout is not None else "inherited",
                         "initial_validation_nll": summary["continuation"]["initial_validation_nll"],
                         "best_new_validation_nll": summary["continuation"]["best_new_validation_nll"],
                         "selected_validation_nll": summary["best_validation_nll"],
                         "retained_initial_model": summary["continuation"]["retained_initial_model"],
                         "updates": summary["global_step"],
                         "samples": json.loads((Path(branch.output_dir) / "best" / "samples.json").read_text())}
        write_json(out / "comparison.json", results)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\nPaired comparison (same source weights, data, learning rate, seed, and update budget):")
    for name, result in results.items():
        print(f"{name}: best new validation NLL={result['best_new_validation_nll']}; "
              f"retained starting model={result['retained_initial_model']}")
    print(f"Read both sets of samples in {out / 'comparison.json'}. "
          "Lower NLL alone does not establish better narrative coherence.")
    return results


def compare_windowing(args):
    return compare_training_variants(args, "windowing")


def compare_depth(args):
    return compare_training_variants(args, "depth")


def compare_training_variants(args, kind):
    """Bounded windowing or depth comparison with the same data/update budget."""
    import gc

    import torch
    from transformers import AutoTokenizer, GPT2Config
    if not args.continue_from or args.resume or args.rebuild or args.smoke_test:
        raise ValueError(f"compare-{kind} requires --continue-from, new output, and no --resume/--rebuild/--smoke-test")
    if args.max_steps <= 0 or args.continuation_dropout is not None:
        raise ValueError("Set --max-steps per branch and inherit the source dropout for this comparison")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new --output-dir for the comparison")
    if kind == "depth":
        config = GPT2Config.from_pretrained(args.continue_from, local_files_only=True)
        if args.grow_layers is None or args.grow_layers <= config.n_layer:
            raise ValueError(f"Set --grow-layers greater than the source depth ({config.n_layer})")
        variants = (("control", None), ("expanded", args.grow_layers))
    else:
        variants = (("fixed", False), ("shifted", True))
    results = {}
    for name, value in variants:
        branch = argparse.Namespace(**vars(args))
        branch.command, branch.output_dir = "train", str(out / name)
        branch.skip_test = True
        if kind == "depth":
            branch.grow_layers = value
        else:
            branch.random_window_offsets = value
        branch.patience = max(args.patience, math.ceil(args.max_steps / args.eval_steps)+1)
        summary = training(branch)
        best = Path(branch.output_dir) / "best"
        tokenizer = AutoTokenizer.from_pretrained(best, local_files_only=True)
        selected = load_continuation(best, tokenizer, summary["dataset_sha256"])
        selected.to("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        passages = []
        for split in ("train", "validation"):
            dataset = BookWindows(Path(args.data_dir) / f"{split}.jsonl", selected.config.n_positions, limit=1)
            for probe in corpus_probe_inputs(dataset, tokenizer, args.sample_tokens, args.seed):
                samples = generate_samples(selected, tokenizer, [probe], args.sample_tokens, args.seed)
                passages.append({"split": split, **probe, "generated": samples})
                print(f"\n[{name} | {split}] PREFIX: {probe['prompt']}", flush=True)
                for sample in samples:
                    print(f"{sample['mode'].upper()}: {sample['continuation']}\n", flush=True)
        write_json(best / "passage_samples.json", passages)
        results[name] = {"random_window_offsets": branch.random_window_offsets,
                         "parameters": summary["parameters"], "layers": summary["run"]["layers"],
                         "depth_growth": summary["continuation"].get("depth_growth"),
                         "initial_validation_nll": summary["continuation"]["initial_validation_nll"],
                         "best_new_validation_nll": summary["continuation"]["best_new_validation_nll"],
                         "selected_validation_nll": summary["best_validation_nll"],
                         "retained_initial_model": summary["continuation"]["retained_initial_model"],
                         "updates": summary["global_step"], "best_model": str(best),
                         "samples": json.loads((best / "samples.json").read_text()), "passage_samples": passages}
        write_json(out / "comparison.json", results)
        del selected, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"\n{kind.upper()} COMPARISON (same source model, target counts, and update budget):")
    if kind == "depth":
        print("The expanded branch uses more compute per update; this is not a matched-compute comparison.")
    print(json.dumps({name: {key: value for key, value in result.items()
                             if key not in {"samples", "passage_samples"}}
                      for name, result in results.items()}, indent=2))
    print("\nMATCHED PASSAGES (include these with the metrics when assessing quality):")
    for split in ("train", "validation"):
        pairs = [(name, next((probe for probe in result["passage_samples"] if probe["split"] == split), None))
                 for name, result in results.items()]
        if any(probe is None for _, probe in pairs):
            continue
        print(f"\n{split.upper()} PREFIX: {pairs[0][1]['prompt']}")
        for name, probe in pairs:
            for sample in probe["generated"]:
                print(f"{name.upper()} / {sample['mode'].upper()}: {sample['continuation']}\n")
    print(f"Read both sets of passages in {out / 'comparison.json'}. "
          "Loss improvement alone is not evidence of coherent generation.")
    return results


def resolve_precision(requested, cpu=False):
    import torch
    cuda = torch.cuda.is_available() and not cpu
    if requested == "auto":
        return ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if cuda else "fp32"
    if requested in {"fp16", "bf16"} and not cuda:
        raise ValueError("fp16/bf16 training requires CUDA; use auto or fp32 on CPU/MPS")
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16")
    return requested


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    """A completion record must never be a partially written JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def training_exposure(dataset, effective_batch, steps, seed):
    """Exact target count, including short book tails and a partial last update."""
    import torch
    if isinstance(dataset, CoherenceDataset):
        result = training_exposure(dataset.base, effective_batch, steps, seed)
        result['target_count_scope'] = 'Replay CE labels only; auxiliary positive/negative forwards and curriculum weights excluded.'
        return result
    if isinstance(dataset, (SceneReplayDataset, WinterReplayDataset)):
        return dataset.exposure(effective_batch, steps, seed)
    lengths = [dataset.scored_targets(i) for i in range(len(dataset))]
    per_epoch = math.ceil(len(lengths)/effective_batch)
    epochs, remainder = divmod(steps, per_epoch)
    generator = torch.Generator().manual_seed(seed+epochs)
    order = torch.randperm(len(lengths), generator=generator).tolist()
    partial = order[:remainder*effective_batch]
    return {"steps": steps, "updates_per_epoch": per_epoch, "complete_epochs": epochs,
            "examples": epochs*len(lengths)+len(partial),
            "scored_targets": epochs*sum(lengths)+sum(lengths[i] for i in partial),
            "targets_per_complete_epoch": sum(lengths)}


def make_capacity_benchmark(data, tokenizer, signature, count=20, seed=42,
                            prefix_min=64, prefix_max=160, generation_tokens=192):
    """Freeze book-balanced validation prompts at paragraph/sentence boundaries.

    Decode/re-encode the complete book only to locate character boundaries.
    Require an exact token round trip, then retain the original token slices.
    References are stored for review only and never passed to generation.
    """
    from bisect import bisect_left, bisect_right
    rng = random.Random(seed)
    candidates = {}
    dataset = BookWindows(Path(data) / "validation.jsonl", 512)
    for book_id, book in zip(dataset.book_ids, dataset.books):
        original = book[1:-1].tolist()  # Prepared books have one boundary EOS at either end.
        text = tokenizer.decode(original, clean_up_tokenization_spaces=False)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        if encoded["input_ids"] != original:
            raise ValueError(f"Cannot locate exact token boundaries in validation book {book_id}")
        starts, ends = zip(*encoded["offset_mapping"])
        sentence_ends = [match.end() for match in re.finditer(r"""[.!?]["'”’)]*(?=\s|$)""", text)]
        paragraphs = [0] + [match.end() for match in re.finditer(r"\n[ \t]*\n+", text)]
        choices = []
        for begin in paragraphs:
            while begin < len(text) and text[begin].isspace():
                begin += 1
            start = bisect_left(starts, begin)
            # Avoid front/back matter and near-end passages without a reference.
            if not len(original)*.05 <= start < len(original)*.9 or starts[start] != begin:
                continue
            minimum = start+prefix_min
            maximum = min(start+prefix_max, len(original)-generation_tokens)
            if minimum > maximum:
                continue
            first = bisect_left(sentence_ends, ends[minimum-1])
            last = bisect_right(sentence_ends, ends[maximum-1])
            options = []
            for boundary in sentence_ends[first:last]:
                end = bisect_right(ends, boundary)
                if prefix_min <= end-start <= prefix_max and ends[end-1] == boundary:
                    options.append(end)
            if not options:
                continue
            end = min(options, key=lambda x: abs(x-start-(prefix_min+prefix_max)//2))
            ids = original[start:end]
            choices.append({"book_id": book_id, "start_token": start+1,
                            "prefix_tokens": len(ids), "input_ids": ids,
                            "prompt": tokenizer.decode(ids, clean_up_tokenization_spaces=False),
                            "reference_ids": original[end:end+generation_tokens],
                            "reference_continuation": tokenizer.decode(original[end:end+generation_tokens])})
        rng.shuffle(choices)
        candidates[book_id] = choices
    order = sorted(candidates)
    rng.shuffle(order)
    prompts = []
    # Round robin prevents the longest books from dominating this qualitative panel.
    while len(prompts) < count:
        added = False
        for book_id in order:
            if candidates[book_id] and len(prompts) < count:
                prompt = candidates[book_id].pop()
                prompt["prompt_id"] = f"p{len(prompts):02d}"
                prompts.append(prompt)
                added = True
        if not added:
            raise ValueError(f"Only {len(prompts)} eligible validation passages; requested {count}")
    return {"version": 1, "dataset_sha256": signature, "selection_seed": seed,
            "split": "validation", "prompts": prompts, "sampling_seeds": [seed, seed+1],
            "generation_tokens": generation_tokens, "training_eval_windows": 1024,
            "decoding": {"temperature": .8, "top_p": .92, "top_k": 40, "repetition_penalty": 1.08,
                         "no_repeat_ngram_size": 0},
            "interpretation": "Frozen before either model trains. Original validation text only. "
            "Read outputs for grammar, character/scene continuity, event progression, and repetition. "
            "An alternative continuation need not match the reference. No automatic coherence score."}


def copied_training_spans(generations, dataset, minimum=32):
    """Find the longest exact training-token span >= minimum, excluding prompts.

    Query-only rolling-hash index: memory scales with generated text, not the
    six-million-token corpus. Every hash match is verified against actual IDs.
    Shorter matches and matches with different token boundaries are not detected.
    This is an overlap diagnostic, not a novelty or coherence verdict.
    """
    base, mask = 1000003, (1 << 64)-1
    power = pow(base, minimum-1, 1 << 64)

    def hashes(tokens):
        if len(tokens) < minimum:
            return
        value = 0
        for token in tokens[:minimum]:
            value = (value*base+int(token)+1) & mask
        yield 0, value
        for end in range(minimum, len(tokens)):
            value = (((value-(int(tokens[end-minimum])+1)*power) & mask)*base+int(tokens[end])+1) & mask
            yield end-minimum+1, value

    queries = defaultdict(list)
    results = [{"matched_tokens": 0, "book_id": None, "training_start_token": None,
                "generated_start_token": None} for _ in generations]
    for index, tokens in enumerate(generations):
        for start, value in hashes(tokens):
            queries[value].append((index, start))
    if not queries:
        return results
    for book_id, book in zip(dataset.book_ids, dataset.books):
        for start, value in hashes(book):
            for index, generated_start in queries.get(value, ()):
                tokens = generations[index]
                if book[start:start+minimum].tolist() != tokens[generated_start:generated_start+minimum]:
                    continue
                left, right = 0, minimum
                while left < min(start, generated_start) and book[start-left-1] == tokens[generated_start-left-1]:
                    left += 1
                while start+right < len(book) and generated_start+right < len(tokens) and book[start+right] == tokens[generated_start+right]:
                    right += 1
                if left+right > results[index]["matched_tokens"]:
                    results[index] = {"matched_tokens": left+right, "book_id": book_id,
                                      "training_start_token": start-left,
                                      "generated_start_token": generated_start-left}
    return results


def write_coherence_sheet(path, records):
    import csv
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    fields = ["sample_id", "prompt", "continuation", "generated_tokens", "ended_with_eos",
              "longest_training_copy_tokens", "grammar_0_2", "continuity_0_2",
              "event_progression_0_2", "nonrepetition_0_2", "notes"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})
    temporary.replace(path)


def score_training_subset(model, tokenizer, dataset, limit=1024, batch_size=8):
    """Fixed training-window NLL with dropout off, without changing the training iterator."""
    import copy

    import numpy as np
    import torch
    fixed = copy.copy(dataset)
    indices = np.linspace(0, len(dataset)-1, min(limit, len(dataset)), dtype=int)
    fixed.windows = [dataset.windows[i] for i in indices]
    fixed.random_offsets = False
    fixed.set_training_epoch(0)
    collator = CausalCollator(tokenizer.pad_token_id)
    total, targets = 0., 0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(fixed), batch_size):
                batch = {k: v.to(model.device) for k, v in
                         collator([fixed[i] for i in range(start, min(start+batch_size, len(fixed)))]).items()}
                logits = model(batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
                values = loss_sums(logits, batch["labels"]).cpu().double().sum(dim=0)
                total += float(values[0])
                targets += int(values[1])
    finally:
        model.train(was_training)
    return {"nll": total/targets, "scored_tokens": targets, "windows": len(fixed), "dropout_disabled": True}


def score_context_use(model, benchmark):
    """Paired teacher-forced validation diagnostic; never a coherence score.

    Keep reference tokens and positional indices fixed, changing only the prefix
    to text from a different validation book. No reference is used as generation
    input. Positive gain can reflect vocabulary/topic cues as well as continuity.
    """
    import torch
    prompts = benchmark["prompts"]
    count = benchmark["context_probe_tokens"]
    records, skipped = [], []
    totals = [0., 0.]
    scored = 0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for i, prompt in enumerate(prompts):
                prefix = prompt["input_ids"]
                reference = prompt["reference_ids"][:count]
                other = next((prompts[(i+j) % len(prompts)] for j in range(1, len(prompts))
                              if prompts[(i+j) % len(prompts)]["book_id"] != prompt["book_id"]
                              and len(prompts[(i+j) % len(prompts)]["input_ids"])
                              + len(prompts[(i+j) % len(prompts)]["reference_ids"]) >= len(prefix)), None)
                if other is None or not reference:
                    skipped.append(prompt["prompt_id"])
                    continue
                # Other-book reference is only source text for the unrelated prefix.
                unrelated = (other["input_ids"]+other["reference_ids"])[:len(prefix)]
                inputs = torch.tensor([prefix+reference, unrelated+reference], device=model.device)
                labels = inputs.clone()
                labels[:, :len(prefix)] = -100
                logits = model(inputs, use_cache=False).logits
                sums = loss_sums(logits, labels).double().cpu()
                assert sums[:, 1].tolist() == [len(reference), len(reference)]
                values = [float(sums[k, 0])/len(reference) for k in (0, 1)]
                records.append({"prompt_id": prompt["prompt_id"], "book_id": prompt["book_id"],
                                "unrelated_prompt_id": other["prompt_id"], "unrelated_book_id": other["book_id"],
                                "prefix_tokens": len(prefix), "reference_tokens": len(reference),
                                "true_prefix_nll": values[0], "unrelated_prefix_nll": values[1],
                                "context_gain_nll": values[1]-values[0]})
                for k in (0, 1):
                    totals[k] += float(sums[k, 0])
                scored += len(reference)
    finally:
        model.train(was_training)
    return {"scored_tokens_per_condition": scored, "paired_prompts": len(records),
            "true_prefix_nll": totals[0]/scored if scored else None,
            "unrelated_prefix_nll": totals[1]/scored if scored else None,
            "context_gain_nll": (totals[1]-totals[0])/scored if scored else None,
            "skipped_prompt_ids": skipped, "pairs": records, "dropout_disabled": True,
            "interpretation": "Positive gain means the real prefix lowers reference NLL relative to an "
                              "unrelated equal-length prefix. Includes lexical/topic cues; neither sustained "
                              "coherence nor a checkpoint-selection rule. Teacher forcing, not generation."}


def run_capacity_review(model, tokenizer, benchmark, training_data, destination, step, nll):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    results, generated_ids = [], []
    start = time.monotonic()
    training_score = score_training_subset(model, tokenizer, getattr(training_data, "original_pool", training_data),
                                          benchmark.get("training_eval_windows", 1024))
    supplementary_score = (score_training_subset(model, tokenizer, training_data.supplementary_pool,
                           min(256, benchmark.get("training_eval_windows", 1024)))
                           if hasattr(training_data, "supplementary_pool") else None)
    for prompt in benchmark["prompts"]:
        for seed in benchmark["sampling_seeds"]:
            sample, = generate_samples(model, tokenizer, [prompt], benchmark["generation_tokens"], seed,
                                       modes=("sample",), include_token_ids=True)
            ids = [token for token in sample["token_ids"]
                   if token not in {tokenizer.eos_token_id, tokenizer.pad_token_id}]
            generated_ids.append(ids)
            results.append({"sample_id": f"{prompt['prompt_id']}-s{seed}", "prompt_id": prompt["prompt_id"],
                            "book_id": prompt["book_id"], "seed": seed, **sample,
                            "generated_tokens": len(ids),
                            "ended_with_eos": bool(sample["token_ids"] and sample["token_ids"][-1] == tokenizer.eos_token_id),
                            **repetition_counts(ids)})
    short_results = []
    if benchmark.get("short_tokens"):
        for result, ids in zip(results, generated_ids):
            short_ids = ids[:benchmark["short_tokens"]]
            short_results.append({**result, "token_ids": short_ids,
                                  "continuation": tokenizer.decode(short_ids, skip_special_tokens=True),
                                  "generated_tokens": len(short_ids),
                                  "ended_with_eos": result["ended_with_eos"] and len(ids) <= len(short_ids),
                                  **repetition_counts(short_ids)})
        generated_ids += [row["token_ids"] for row in short_results]
    for result, match in zip(results+short_results, copied_training_spans(generated_ids, training_data)):
        result["training_copy"] = match
        result["longest_training_copy_tokens"] = match["matched_tokens"]
    context_use = score_context_use(model, benchmark) if benchmark.get("context_probe_tokens") else None
    reranked = []
    if benchmark.get('reranked_prompts'):
        for prompt in benchmark['prompts'][:benchmark['reranked_prompts']]:
            for seed in benchmark['sampling_seeds']:
                sample, = generate_samples(model, tokenizer, [prompt], benchmark['generation_tokens'], seed,
                                            modes=('reranked',), include_token_ids=True)
                content = [t for t in sample['token_ids'] if t not in {tokenizer.pad_token_id, tokenizer.eos_token_id}]
                reranked.append({**sample, 'sample_id': f"{prompt['prompt_id']}-s{seed}-reranked",
                                 'prompt_id': prompt['prompt_id'], 'book_id': prompt['book_id'],
                                 'seed': seed, 'generated_tokens': len(content),
                                 'ended_with_eos': bool(sample['token_ids'] and sample['token_ids'][-1] == tokenizer.eos_token_id),
                                 **repetition_counts(content)})
        for sample, match in zip(reranked, copied_training_spans([s['token_ids'] for s in reranked], training_data)):
            sample['training_copy'] = match
            sample['longest_training_copy_tokens'] = match['matched_tokens']
        write_coherence_sheet(destination/'reranked_review.csv', reranked)
    report = {"step": step, "validation_nll": nll, "benchmark_sha256": json_digest(benchmark),
              "test_evaluated": False, "fixed_training_subset": training_score,
              "fixed_supplementary_subset": supplementary_score,
              "context_use": context_use,
              "reranked_samples": reranked,
              "samples": results, "runtime_seconds": time.monotonic()-start,
              "copy_search": "Exact generated-only token spans >=32, searched within training books. "
              "Zero means no match detected by this rule, not proof of originality.",
              "copy_reference_scope": getattr(training_data, "copy_reference_scope", "Original training books"),
              "review_rubric": {"0": "clear failure", "1": "mixed or uncertain", "2": "sustained success"},
              "acceptance": "Manual review required; loss, low repetition, and copying are not coherence scores."}
    if benchmark.get('continuity_pairs'):
        report['continuity_preferences'] = score_continuity_pairs(
            model, benchmark['continuity_pairs'], tokenizer.pad_token_id)
    write_coherence_sheet(destination / "review.csv", results)
    if short_results:
        report["short_samples"] = short_results
        report["short_panel_note"] = "First tokens of the same continuations; not additional independent samples."
        write_coherence_sheet(destination / "short_review.csv", short_results)
    atomic_json(destination / "coherence.json", report)  # Completion marker written last.
    print(f"\n[coherence | step {step}] {len(results)} continuations: {destination / 'coherence.json'}", flush=True)
    for result in results[:2]:
        print_sample(result, f"seed {result['seed']}")
    return report


def capacity_callback(args, tokenizer, training_data):
    from transformers import TrainerCallback
    out = Path(args.output_dir)
    skip_path = out / "optimizer_skips.json"

    class CapacityCallback(TrainerCallback):
        def skipped_count(self, step):
            return len([s for s in json.loads(skip_path.read_text()) if s <= step]) if skip_path.exists() else 0

        def record(self, model, state, force=False):
            due = state.global_step % args.benchmark_steps == 0 or state.global_step == state.max_steps
            if not due or state.global_step == 0:
                return
            destination = out / "reviews" / f"step-{state.global_step}"
            if not force and (destination / "coherence.json").exists():
                return
            model.save_pretrained(destination / "model", safe_serialization=True)
            tokenizer.save_pretrained(destination / "model")
            nll = next((row["eval_nll"] for row in reversed(state.log_history)
                        if row.get("step") == state.global_step and "eval_nll" in row), None)
            atomic_json(destination / "model" / "training_summary.json", {
                "dataset_sha256": args.capacity_benchmark["dataset_sha256"], "global_step": state.global_step,
                "validation_nll": nll, "parameters": model.num_parameters(), "test": {},
                "initialization": "project_checkpoint" if (getattr(args, "scene_recipe", None)
                    or getattr(args, "winter_recipe", None)) else "random",
                "winter_recipe": getattr(args, "winter_recipe", None),
                "benchmark_sha256": args.benchmark_sha256})
            if getattr(args, "winter_recipe", None):
                atomic_json(destination / "model" / "supplementary_sources.json", args.winter_manifest)
            run_capacity_review(model, tokenizer, args.capacity_benchmark, training_data,
                                destination, state.global_step, nll)

        def on_train_begin(self, training_args, state, control, **kwargs):
            prior = json.loads(skip_path.read_text()) if skip_path.exists() else []
            atomic_json(skip_path, [s for s in prior if s <= state.global_step])
            # Recover a checkpoint saved immediately before a review was interrupted.
            self.record(kwargs["model"], state)

        def on_step_end(self, training_args, state, control, **kwargs):
            if getattr(kwargs["optimizer"], "step_was_skipped", False):
                prior = json.loads(skip_path.read_text()) if skip_path.exists() else []
                atomic_json(skip_path, sorted(set(prior+[state.global_step])))
            if state.global_step % args.benchmark_steps == 0 or state.global_step == state.max_steps:
                # Include the final partial interval in evaluation and persistence.
                control.should_evaluate = control.should_save = True
            return control

        def on_save(self, training_args, state, control, **kwargs):
            self.record(kwargs["model"], state, force=True)

    return CapacityCallback()


def compare_capacity(args):
    """Fresh architectures; fixed target exposure; resumable optimizers; frozen prose review."""
    import gc
    import os

    import torch
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This comparison supports one training process/device so batch accounting stays exact")
    if not args.cpu and not args.smoke_test and torch.cuda.is_available() and torch.cuda.device_count() != 1:
        raise ValueError("Expose one CUDA device (CUDA_VISIBLE_DEVICES) for this matched-batch experiment")
    if args.continue_from or args.grow_layers or args.corpus or args.rebuild:
        raise ValueError("Capacity comparison starts fresh on the original source list; no continuation/subset/rebuild")
    if args.resume not in {None, "latest"}:
        raise ValueError("Use --resume without a checkpoint path to resume the paired experiment")
    if args.train_limit or args.eval_limit:
        raise ValueError("Use --smoke-test for a limited check; the real comparison uses the complete splits")
    if (args.block_size != 512 or args.vocab_size != 8192
            or args.batch_size*args.accumulation != 32 or args.min_success_rate != 1):
        raise ValueError("Capacity comparison requires context 512, vocab 8192, effective batch 32, and all sources")
    if not args.random_window_offsets:
        raise ValueError("Capacity comparison requires the agreed shifted training windows")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Use a new output directory or --resume (also after --plan-only)")
    if args.resume and not (out / "experiment.json").exists():
        raise ValueError("No saved experiment plan to resume")
    prepare(args)
    data = Path(args.data_dir)
    signature = dataset_signature(data)
    manifest = json.loads((data / "manifest.json").read_text())
    if any(book["url"] not in sources for book in manifest):
        raise ValueError("Data contain sources outside the approved Christmas corpus")
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    if len(tokenizer) != 8192:
        raise ValueError("The paired architectures require the existing 8192-token vocabulary")
    variants = [("14m", 6, 384, 6), ("42m", 12, 512, 8)]
    prefix_min, prefix_max = 64, 160
    if args.smoke_test:
        variants = [("tiny-control", 2, 64, 2), ("tiny-larger", 3, 96, 3)]
        args.cpu, args.precision, args.block_size = True, "fp32", 128
        args.batch_size, args.accumulation, args.max_steps = 2, 1, 3
        args.train_limit, args.eval_limit, args.eval_steps = 16, 8, 2
        args.sample_steps, args.sample_tokens = 3, 12
        args.benchmark_steps, args.benchmark_prompts, args.benchmark_tokens = 2, 2, 12
        prefix_min, prefix_max = 8, 48
    args.precision = resolve_precision(args.precision, args.cpu)
    args.fixed_budget, args.skip_test = True, True
    training_data = BookWindows(data / "train.jsonl", args.block_size, args.train_limit,
                                random_offsets=True, seed=args.seed)
    effective_batch = args.batch_size*args.accumulation
    steps = args.max_steps if args.max_steps > 0 else math.ceil(
        math.ceil(len(training_data)/effective_batch)*args.epochs)
    if steps < args.eval_steps:
        raise ValueError("Budget is shorter than --eval-steps; lower the evaluation interval")
    benchmark = make_capacity_benchmark(data, tokenizer, signature, args.benchmark_prompts, args.seed,
                                       prefix_min, prefix_max, args.benchmark_tokens)
    benchmark_hash = json_digest(benchmark)
    architectures = []
    for name, layers, width, heads in variants:
        with torch.device("meta"):
            parameter_count = GPT2LMHeadModel(GPT2Config(
                vocab_size=len(tokenizer), n_positions=args.block_size, n_ctx=args.block_size,
                n_layer=layers, n_embd=width, n_head=heads, n_inner=4*width)).num_parameters()
        architectures.append({"name": name, "layers": layers, "width": width, "heads": heads,
                              "parameters": parameter_count})
    shared = {key: getattr(args, key) for key in (
        "seed", "model_seed", "block_size", "dropout", "attention_dropout", "batch_size", "accumulation",
        "learning_rate", "warmup_ratio", "epochs", "max_steps", "eval_steps", "sample_steps",
        "sample_tokens", "benchmark_steps", "precision", "smoke_test", "train_limit", "eval_limit")}
    plan = {"version": 1, "dataset_sha256": signature, "benchmark_sha256": benchmark_hash,
            "architectures": architectures, "shared": shared, "initialization": "random for both branches",
            "planned_exposure_per_branch": training_exposure(training_data, effective_batch, steps, args.seed),
            "warmup_updates": math.ceil(steps*args.warmup_ratio),
            "final_learning_rate": args.learning_rate*.1, "test_evaluated": False,
            "matched_compute": False, "early_stopping": False,
            "rationale": {
                "architecture": "Compare 6x384 with 12x512; both use 64-dimensional attention heads, 4x MLPs, "
                "tied embeddings, and GPT-2's residual-projection initialization. Fresh weights, no depth growth.",
                "corpus": "Same prepared books, overlap-group splits, train-only BPE, shuffling, and epoch offsets. "
                "No additional text, synthetic stories, changed sampling weights, or chapter rewriting.",
                "regularization": "0.05 dropout in embedding, residual, and attention paths by default; AdamW "
                "weight decay 0.1, excluding bias/norm parameters through Trainer. Experimental, not proven optimal.",
                "optimization": "Same AdamW (0.9,0.95), epsilon 1e-8, gradient clip 1.0; default peak LR 3e-4. "
                "One warmup/cosine schedule ending at 10% of peak; no optimizer restarts between epochs.",
                "budget": "40 epochs by default. Early stopping disabled to keep exposures matched. "
                "Change the budget before starting both branches. Repetition is not additional unique evidence.",
                "precision": "BF16 on supported CUDA GPUs, otherwise CUDA FP16 with scaling; FP32 on CPU/MPS. "
                "AMP skipped steps are reported. Exact bitwise replay across hardware/software is not promised.",
                "selection": "Retain best validation NLL and periodic review weights. Human coherence review "
                "is required; there is no automatic semantic winner. Larger model costs more per update."}}
    if args.smoke_test:
        plan["rationale"]["architecture"] = (
            "Two distinct tiny architectures, 2x64 and 3x96, with 32-dimensional heads. "
            "Execution verification only; this is not the real 14M/42M comparison.")
    if args.resume:
        if json.loads((out / "experiment.json").read_text()) != plan:
            raise ValueError("Saved experiment settings/data differ; restore them or use a new output directory")
        if json.loads((out / "benchmark.json").read_text()) != benchmark:
            raise ValueError("Frozen benchmark differs; refusing to change the evaluation panel on resume")
    else:
        atomic_json(out / "benchmark.json", benchmark)
        atomic_json(out / "experiment.json", plan)
    print("\nCAPACITY PLAN (fresh models; matched examples/targets; different compute):", flush=True)
    print(json.dumps({key: value for key, value in plan.items() if key != "rationale"}, indent=2), flush=True)
    if args.plan_only:
        print(f"Plan and frozen prompts: {out}. Add --resume to start this saved plan.", flush=True)
        return plan
    results = {}
    for architecture in architectures:
        name = architecture["name"]
        branch = argparse.Namespace(**vars(args))
        branch.command, branch.output_dir = "train", str(out / name)
        branch.layers, branch.width, branch.heads = (architecture[k] for k in ("layers", "width", "heads"))
        branch.capacity_benchmark, branch.benchmark_sha256 = benchmark, benchmark_hash
        branch.resume = "latest" if args.resume and (Path(branch.output_dir) / "run_config.json").exists() else None
        best = Path(branch.output_dir) / "best"
        complete = best / "training_summary.json"
        if args.resume and complete.exists() and (best / "coherence.json").exists():
            summary = json.loads(complete.read_text())
            if (summary["dataset_sha256"] != signature or summary["global_step"] != steps
                    or summary["parameters"] != architecture["parameters"]
                    or summary["run"].get("benchmark_sha256") != benchmark_hash):
                raise ValueError(f"Completed branch metadata do not match the plan: {name}")
            print(f"Reusing completed branch {name}; optimizer will not restart.", flush=True)
        else:
            summary = training(branch)
        results[name] = {"parameters": summary["parameters"], "global_step": summary["global_step"],
                         "best_validation_nll": summary["best_validation_nll"],
                         "best_validation_step": summary["best_validation_step"],
                         "training_exposure": summary["training_exposure"],
                         "optimizer_skipped_steps": summary["optimizer_skipped_steps"],
                         "applied_optimizer_updates": summary["applied_optimizer_updates"],
                         "best_model": str(best), "coherence_report": str(best / "coherence.json"),
                         "final_review": str(Path(branch.output_dir) / "reviews" / f"step-{steps}")}
        atomic_json(out / "comparison.json", {
            "complete": len(results) == len(architectures), "benchmark_sha256": benchmark_hash,
            "test_evaluated": False, "matched_compute": False, "branches": results,
            "selected_checkpoints_may_differ_in_step": True,
            "coherence_winner": None, "interpretation": "Review prose; lower NLL is not a coherence verdict."})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if len({json_digest(result["training_exposure"]) for result in results.values()}) != 1:
        raise RuntimeError("Branch target exposure differs; comparison is invalid")
    # Mixed, opaque IDs allow manual review without seeing the architecture label.
    panels, key = {"best": [], "final": []}, {}
    for name, result in results.items():
        for selection, report_path, checkpoint in (
                ("best", Path(result["coherence_report"]), result["best_model"]),
                ("final", Path(result["final_review"]) / "coherence.json",
                 str(Path(result["final_review"]) / "model"))):
            for sample in json.loads(report_path.read_text())["samples"]:
                sample_id = hashlib.sha256(
                    f"{benchmark_hash}:{name}:{selection}:{sample['sample_id']}".encode()).hexdigest()[:16]
                panels[selection].append({**sample, "sample_id": sample_id})
                key[sample_id] = {"branch": name, "selection": selection,
                                  "sample_id": sample["sample_id"], "checkpoint": checkpoint}
    rng = random.Random(args.seed)
    for selection, filename in (("best", "blind_review.csv"), ("final", "blind_final_review.csv")):
        rng.shuffle(panels[selection])
        if not (args.resume and (out / filename).exists()):
            write_coherence_sheet(out / filename, panels[selection])
    atomic_json(out / "review_key.json", key)
    print("\nCOMPLETED CAPACITY COMPARISON:", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print(f"Read/rate {out / 'blind_review.csv'} (best NLL) and blind_final_review.csv (same final step); "
          "decode branch names with review_key.json afterward. "
          "The test set remains reserved. Smoke runs validate execution only.", flush=True)
    return results


def scene_sentence_ends(text):
    """Conservative boundary heuristic, not a syntactic/semantic parser."""
    abbreviations = {"mr", "mrs", "ms", "dr", "st", "rev", "prof", "jr", "sr"}
    ends = []
    for match in re.finditer(r'''[.!?]["'”’)]*(?=\s|$)''', text):
        before = re.search(r"([A-Za-z]+)$", text[:match.start()])
        if match.group()[0] == "." and before and (
                before[1].lower() in abbreviations or len(before[1]) == 1):
            continue
        ends.append(match.end())
    return ends


def extract_scene_candidates(dataset, tokenizer, minimum=192, maximum=448, per_book=16):
    """Rank exact, nonoverlapping training-book excerpts by simple surface features.

    These are CANDIDATES, not certified complete scenes. Neither punctuation nor
    readability establishes semantic closure. Automatic cuts never acquire EOS.
    A CSV review can remove candidates and explicitly mark natural scene ends.
    """
    from bisect import bisect_left, bisect_right
    from collections import Counter

    texts = [tokenizer.decode(book[1:-1].tolist(), clean_up_tokenization_spaces=False)
             for book in dataset.books]
    counts = Counter(word.lower() for text in texts for word in re.findall(r"[A-Za-z]+", text))
    selected, seen = [], set()
    rejected = Counter()
    for book_id, book, text in zip(dataset.book_ids, dataset.books, texts):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        if encoded["input_ids"] != book[1:-1].tolist():
            raise ValueError(f"Scene extraction cannot round-trip training book {book_id}")
        starts, ends = zip(*encoded["offset_mapping"])
        paragraphs = [(m.start(), m.end()) for m in re.finditer(r"\S(?:[^\n]|\n(?![ \t]*\n))*", text)]
        options = []
        for paragraph_index, (begin, _) in enumerate(paragraphs):
            # Avoid headings/front matter and obvious unanchored dialogue or anaphora.
            first = text[begin:paragraphs[paragraph_index][1]].strip()
            if not first or first[0] in '\"\'“‘' or re.match(
                    r"(?i)^(he|she|they|it|his|her|their|then|but|and|so|this|that)\b", first):
                rejected["unanchored_opening"] += 1
                continue
            start = bisect_left(starts, begin)
            if start >= len(starts) or starts[start] != begin or start < len(starts)*.03:
                continue
            for _, raw_end in paragraphs[paragraph_index:paragraph_index+12]:
                finish = raw_end
                while finish > begin and text[finish-1].isspace():
                    finish -= 1
                end = bisect_right(ends, finish)
                if end-start > maximum:
                    break
                if end-start < minimum or end == 0 or ends[end-1] != finish:
                    continue
                passage = text[begin:finish]
                blocks = re.split(r"\n\s*\n", passage)
                if any(len(b.split()) < 5 or b.strip().isupper() or re.match(
                        r"(?i)^\s*(chapter\b|book [ivx]+\b|\[illustration|contents\b|footnote\b)", b)
                       or (b.count("\n") >= 3 and len(b.split())/(b.count("\n")+1) < 8)
                       for b in blocks):
                    rejected["heading_or_verse"] += 1
                    continue
                boundaries = scene_sentence_ends(passage)
                if len(boundaries) < 4 or boundaries[-1] != len(passage):
                    rejected["sentence_boundaries"] += 1
                    continue
                words = re.findall(r"[A-Za-z]+", passage)
                sentence_lengths = [len(re.findall(r"[A-Za-z]+", passage[a:b]))
                                    for a, b in zip([0]+boundaries[:-1], boundaries)]
                mean_length = sum(sentence_lengths)/len(sentence_lengths)
                rare_fraction = sum(counts[w.lower()] < 20 for w in words)/max(1, len(words))
                # Capitalized interior words are only a proxy for character/setting complexity.
                names = set(re.findall(r"(?<=[a-z,;] )\b[A-Z][a-z]{2,}\b", passage))
                if mean_length > 22 or max(sentence_lengths) > 50 or rare_fraction > .12 or len(names) > 6:
                    rejected["complexity_proxy"] += 1
                    continue
                prefix_options = []
                for boundary in boundaries[:-1]:
                    position = bisect_right(ends, begin+boundary)
                    if (48 <= position-start <= 128 and end-position >= 64
                            and ends[position-1] == begin+boundary):
                        prefix_options.append(position-start)
                if not prefix_options:
                    rejected["prefix_length"] += 1
                    continue
                prefix = min(prefix_options, key=lambda x: abs(x-96))
                ids = book[start+1:end+1].tolist()
                assert tokenizer.decode(ids, clean_up_tokenization_spaces=False) == passage
                content_hash = hashlib.sha256(passage.encode()).hexdigest()
                score = mean_length + 100*rare_fraction + 2*len(names)
                options.append({"scene_id": f"{book_id}-{start+1}-{end+1}", "book_id": book_id,
                                "start_token": start+1, "end_token": end+1, "prefix_tokens": prefix,
                                "text_sha256": content_hash, "text": passage, "input_ids": ids,
                                "natural_end": False, "original_book_end": end == len(starts),
                                "score": round(score, 6), "mean_sentence_words": round(mean_length, 3),
                                "rare_word_fraction": round(rare_fraction, 4), "name_proxy_count": len(names)})
        kept = []
        for row in sorted(options, key=lambda r: (r["score"], r["start_token"], r["end_token"])):
            if row["text_sha256"] in seen or any(
                    row["start_token"] < other["end_token"] and other["start_token"] < row["end_token"]
                    for other in kept):
                continue
            kept.append(row)
            seen.add(row["text_sha256"])
            if len(kept) == per_book:
                break
        selected.extend(kept)
    return selected, {"method": "automatic_surface_heuristics_not_semantic_scene_certification",
                      "candidates": len(selected), "books": len({r["book_id"] for r in selected}),
                      "unique_excerpt_tokens": sum(len(r["input_ids"]) for r in selected),
                      "rejections": dict(rejected), "automatic_scene_eos": False}


def write_scene_sheet(path, records):
    import csv
    fields = ["scene_id", "book_id", "start_token", "end_token", "text_sha256", "include",
              "natural_end", "prefix_tokens", "score", "text"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({k: (1 if k == "include" else 0 if k == "natural_end" else row[k]) for k in fields})


def apply_scene_review(records, path):
    """Only inclusion and a natural-end annotation may change; never source text."""
    import csv
    lookup = {row["scene_id"]: row for row in records}
    selected, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            scene_id = row["scene_id"]
            if scene_id not in lookup or scene_id in seen:
                raise ValueError(f"Unknown/duplicate scene ID in review: {scene_id}")
            seen.add(scene_id)
            original = lookup[scene_id]
            for field in ("book_id", "start_token", "end_token", "text_sha256", "prefix_tokens", "text"):
                if str(original[field]) != row[field]:
                    raise ValueError(f"Scene review changed original content or boundaries: {scene_id}/{field}")
            if row["include"] not in {"0", "1"} or row["natural_end"] not in {"0", "1"}:
                raise ValueError("Scene CSV include/natural_end must be 0 or 1")
            if row["include"] == "1":
                selected.append({**original, "natural_end": row["natural_end"] == "1"})
    if not selected:
        raise ValueError("Scene review selects no training passages")
    return selected


class SceneReplayDataset:
    """50% scene examples / 50% original windows, before token-loss weighting.

    Each epoch contains every original replay window once. Scene slots cycle
    through an order interleaved across books. Prefixes are context only; original and
    explicitly reviewed natural ends are the only additional EOS targets.
    """
    def __init__(self, replay, records, eos_id, seed=42):
        if not records:
            raise ValueError("Scene mixture needs at least one selected training excerpt")
        self.replay, self.records, self.eos_id, self.seed = replay, records, eos_id, seed
        self.block_size, self.random_offsets = replay.block_size, replay.random_offsets
        train_books = dict(zip(replay.book_ids, replay.books))
        self.scene_rows = []
        for row in records:
            book = train_books.get(row["book_id"])
            if book is None or book[row["start_token"]:row["end_token"]].tolist() != row["input_ids"]:
                raise ValueError("Scene is not an exact slice of a training book")
            ids = list(row["input_ids"])
            if row["original_book_end"]:
                if row["end_token"] != len(book)-1 or int(book[-1]) != eos_id:
                    raise ValueError("Invalid original-book ending annotation")
            if row["natural_end"] or row["original_book_end"]:
                ids.append(eos_id)
            prefix = row["prefix_tokens"]
            if not 1 <= prefix < len(ids) or len(ids) > self.block_size:
                raise ValueError("Scene prefix/target does not fit the source model's context")
            self.scene_rows.append({"input_ids": ids, "labels": [-100]*prefix+ids[prefix:]})
        self.set_training_epoch(0)

    def __len__(self):
        return 2*len(self.replay)

    def set_training_epoch(self, epoch):
        self.epoch = int(epoch)
        self.replay.set_training_epoch(epoch)
        rng = random.Random(self.seed+self.epoch)
        books = defaultdict(list)
        for i, row in enumerate(self.records):
            books[row["book_id"]].append(i)
        for rows in books.values():
            rng.shuffle(rows)
        order = list(books)
        rng.shuffle(order)
        self.scene_order = []
        while any(books.values()):
            for book in order:
                if books[book]:
                    self.scene_order.append(books[book].pop())

    def __getitem__(self, index):
        slot, is_replay = divmod(index, 2)
        if is_replay:
            return self.replay[slot]
        return self.scene_rows[self.scene_order[slot % len(self.scene_order)]]

    def exposure(self, effective_batch, steps, seed):
        import torch
        previous_epoch = self.epoch
        per_epoch = math.ceil(len(self)/effective_batch)
        result = {"steps": steps, "updates_per_epoch": per_epoch, "complete_epochs": steps//per_epoch,
                  "examples": 0, "scored_targets": 0, "scene_examples": 0, "scene_targets": 0,
                  "replay_examples": 0, "replay_targets": 0}
        remaining = steps
        try:
            for epoch in range(math.ceil(steps/per_epoch)):
                self.set_training_epoch(epoch)
                count = min(remaining, per_epoch)*effective_batch
                order = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed+epoch)).tolist()
                for index in order[:count]:
                    row = self[index]
                    targets = sum(t != -100 for t in row["labels"][1:]) if "labels" in row else len(row["input_ids"])-1
                    kind = "replay" if index % 2 else "scene"
                    result[kind+"_examples"] += 1
                    result[kind+"_targets"] += targets
                    result["examples"] += 1
                    result["scored_targets"] += targets
                remaining -= min(remaining, per_epoch)
        finally:
            self.set_training_epoch(previous_epoch)
        result["scene_fraction_of_scored_targets"] = result["scene_targets"]/max(1, result["scored_targets"])
        return result


def finetune_scenes(args):
    """Bounded option-1 pilot; all text and tokenization come from the saved corpus."""
    import gc
    import os

    import torch
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, set_seed

    if args.corpus or args.rebuild or args.grow_layers or args.continuation_dropout is not None:
        raise ValueError("finetune-scenes keeps the original corpus, architecture, tokenizer, and dropout")
    if args.resume not in {None, "latest"}:
        raise ValueError("Use --resume without a checkpoint path for the whole scene experiment")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
            torch.cuda.is_available() and not args.cpu and torch.cuda.device_count() != 1):
        raise ValueError("Expose one training device/process for reproducible scene sampling")
    if not (Path(args.data_dir)/"report.json").is_file():
        raise ValueError("Restore the source run's prepared data/ directory; scene continuation never rebuilds it")
    out, data = Path(args.output_dir), Path(args.data_dir)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Use a new scene output directory or --resume (also after --plan-only)")
    if args.resume and not (out/"experiment.json").is_file():
        raise ValueError("No frozen scene experiment to resume")
    if args.max_steps <= 0 or args.max_steps < args.eval_steps:
        raise ValueError("Set a positive scene pilot budget at least as large as --eval-steps")
    if args.train_limit or args.eval_limit:
        raise ValueError("Use --smoke-test for execution checks; real scene runs use full train/validation splits")
    prepare(args)
    signature = dataset_signature(data)
    manifest = json.loads((data/"manifest.json").read_text())
    if any(row["url"] not in sources for row in manifest):
        raise ValueError("Prepared data contain sources outside the original Christmas list")
    tokenizer = AutoTokenizer.from_pretrained(data/"tokenizer", local_files_only=True)
    original = BookWindows(data/"train.jsonl", 512, random_offsets=True, seed=args.seed)
    source = Path(args.continue_from or ROOT/"runs/christmas-capacity/14m/best").resolve()
    snapshot = out/"starting_model"
    if out.resolve() == source or out.resolve() in source.parents:
        raise ValueError("Scene output must not contain or replace the source checkpoint")
    if args.resume:
        prior = json.loads((out/"experiment.json").read_text())
        model = load_continuation(snapshot, tokenizer, signature)
        source_description = prior["source"]
    elif args.smoke_test:
        set_seed(args.model_seed)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tokenizer), n_positions=512, n_ctx=512, n_layer=1, n_embd=32,
            n_head=2, n_inner=128, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            resid_pdrop=.05, embd_pdrop=.05, attn_pdrop=.05))
        source_description = {"kind": "random_tiny_execution_check", "seed": args.model_seed}
    else:
        if not source.is_dir():
            raise ValueError(f"Source model not found: {source}. Set --continue-from to the actual "
                             "14m/best directory containing model.safetensors. Reports alone are not weights.")
        model = load_continuation(source, tokenizer, signature)
        if (model.config.n_layer, model.config.n_embd, model.config.n_head, model.config.n_positions) != (6, 384, 6, 512):
            raise ValueError("This option-1 pilot requires the existing 6x384, 6-head, context-512 model")
        source_description = {"kind": "existing_project_14m", "path": str(source),
                              "parameters": model.num_parameters()}
    args.block_size = model.config.n_positions
    weight_digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        weight_digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}".encode())
        weight_digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if args.smoke_test:
        args.cpu, args.precision = True, "fp32"
        args.batch_size, args.accumulation, args.max_steps = 2, 1, 3
        args.train_limit, args.eval_limit = 16, 8
        args.eval_steps, args.sample_steps, args.benchmark_steps = 2, 2, 2
        args.sample_tokens, args.benchmark_tokens, args.benchmark_prompts = 12, 12, 2
    args.precision = resolve_precision(args.precision, args.cpu)
    candidates, extraction = extract_scene_candidates(
        original, tokenizer, args.scene_min_tokens, args.scene_max_tokens, args.scenes_per_book)
    records = apply_scene_review(candidates, args.scene_review) if args.scene_review else candidates
    if len(records) < (1 if args.smoke_test else 128):
        raise ValueError(f"Only {len(records)} scenes selected; need 128 for the real pilot. "
                         "Review extraction settings instead of silently training on a tiny repeated subset.")
    selection = {"dataset_sha256": signature, "extraction": extraction,
                 "selection": "review_csv" if args.scene_review else "automatic_candidates",
                 "records": records}
    benchmark = make_capacity_benchmark(data, tokenizer, signature, args.benchmark_prompts,
                                       args.seed, generation_tokens=args.benchmark_tokens)
    original_benchmark_hash = json_digest(benchmark)
    benchmark["short_tokens"] = min(96, args.benchmark_tokens)
    if args.smoke_test:
        benchmark["training_eval_windows"] = 16
    benchmark_hash = json_digest(benchmark)
    recipe = {"version": 1, "selection_sha256": json_digest(selection),
              "scene_example_fraction": .5, "prefix_loss": "masked",
              "scene_eos": "original_book_end_or_reviewed_natural_end_only"}
    settings = {k: getattr(args, k) for k in (
        "seed", "model_seed", "batch_size", "accumulation", "learning_rate", "warmup_ratio", "max_steps",
        "eval_steps", "sample_steps", "sample_tokens", "benchmark_steps", "precision", "smoke_test",
        "scene_min_tokens", "scene_max_tokens", "scenes_per_book", "train_limit", "eval_limit")}
    replay = BookWindows(data/"train.jsonl", args.block_size, args.train_limit,
                         random_offsets=True, seed=args.seed)
    mixture = SceneReplayDataset(replay, records, tokenizer.eos_token_id, args.seed)
    plan = {"version": 1, "source": source_description, "starting_weights_sha256": weight_digest.hexdigest(),
            "source_configuration": {k: getattr(model.config, k) for k in (
                "n_layer", "n_embd", "n_head", "n_positions", "resid_pdrop", "embd_pdrop", "attn_pdrop")},
            "dataset_sha256": signature,
            "benchmark_sha256": benchmark_hash, "original_full_benchmark_sha256": original_benchmark_hash,
            "settings": settings, "recipe": recipe, "test_evaluated": False,
            "matched_compute": False, "matched_target_counts": False, "coherence_winner": None,
            "selected_scenes": len(records), "selected_books": len({r["book_id"] for r in records}),
            "reviewed_natural_ends": sum(r["natural_end"] for r in records),
            "planned_exposure": {
                "control": training_exposure(replay, args.batch_size*args.accumulation, args.max_steps, args.seed),
                "scenes": training_exposure(mixture, args.batch_size*args.accumulation, args.max_steps, args.seed)},
            "rationale": {
                "source": "Reuse this project's best 14M weights and train-only tokenizer; no external pretraining.",
                "selection": "Exact nonoverlapping paragraph-aligned excerpts, 192-448 tokens by default. "
                "Prefer shorter sentences, fewer rare words and capitalized-name proxies. These heuristics "
                "do not certify semantic self-containment. Inspect scene_candidates.csv; optionally supply a reviewed copy.",
                "mixture": "Half of examples are scene prefixes plus original continuations; half are original "
                "shifted windows. Each mixture epoch includes every replay window once. Scene prefixes are "
                "context only. Token-weighted loss gives scenes less than half of targets; exact counts are reported.",
                "endings": "Automatic paragraph cuts never receive invented EOS labels. Only original book ends "
                "and natural ends explicitly annotated in the review CSV teach an excerpt to stop.",
                "optimization": "Fresh AdamW, peak 5e-5 by default (one sixth of pretraining peak), betas .9/.95, "
                "weight decay .1, clip 1. Inherit dropout. 3% warmup, cosine to 10% of peak, 1000 updates. "
                "A bounded starting hypothesis, not optimized or proven to yield coherent prose.",
                "control": "Same starting weights, optimizer settings and update count; ordinary training versus "
                "the full scene intervention. Target counts and compute differ because scenes mask prefixes.",
                "review": "Retain unchanged baseline, best full-validation-NLL model, and all review snapshots. "
                "Read 192-token continuations plus their first 96 tokens. Lower NLL is not a coherence verdict."}}
    if args.resume:
        if prior != plan or json.loads((out/"scene_selection.json").read_text()) != selection:
            raise ValueError("Frozen scene settings, source annotations, or dataset changed; use a new experiment")
        if json.loads((out/"benchmark.json").read_text()) != benchmark:
            raise ValueError("Frozen scene benchmark changed")
    else:
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(snapshot, safe_serialization=True)
        tokenizer.save_pretrained(snapshot)
        atomic_json(snapshot/"provenance.json", {"dataset_sha256": signature})
        atomic_json(out/"scene_selection.json", selection)
        write_scene_sheet(out/"scene_candidates.csv", candidates)
        atomic_json(out/"benchmark.json", benchmark)
        atomic_json(out/"experiment.json", plan)
    del model, original, mixture, replay
    gc.collect()
    print("\nSCENE PILOT PLAN:", flush=True)
    print(json.dumps({k: v for k, v in plan.items() if k != "rationale"}, indent=2), flush=True)
    print("Selections are original-text candidates; automatic filters do not prove scene coherence. "
          f"Review {out/'scene_candidates.csv'}.", flush=True)
    if args.plan_only:
        print("Plan saved. Add --resume to run unchanged; to use edited annotations, start a NEW output "
              "with --scene-review pointing to the edited CSV.", flush=True)
        return plan
    results = {}
    for name in ("control", "scenes"):
        branch = argparse.Namespace(**vars(args))
        branch.output_dir, branch.command = str(out/name), "train"
        branch.continue_from = str(snapshot.resolve())
        branch.fixed_budget, branch.skip_test, branch.random_window_offsets = True, True, True
        branch.capacity_benchmark, branch.benchmark_sha256 = benchmark, benchmark_hash
        branch.scene_recipe, branch.scene_records = {**recipe, "treatment": name == "scenes"}, records
        branch.resume = "latest" if args.resume and (out/name/"run_config.json").exists() else None
        if branch.resume and latest_complete_checkpoint(out/name, require_scaler=args.precision == "fp16"):
            # Restore the stage's recorded baseline and optimizer. A fresh
            # --continue-from would discard initial_validation_nll on resume.
            branch.continue_from = None
        complete = out/name/"best/training_summary.json"
        if branch.resume and complete.exists() and (out/name/"best/coherence.json").exists():
            summary = json.loads(complete.read_text())
            if (summary["global_step"] != args.max_steps or summary["dataset_sha256"] != signature
                    or summary["run"].get("scene_recipe") != branch.scene_recipe):
                raise ValueError(f"Completed scene branch does not match frozen plan: {name}")
        else:
            summary = training(branch)
        results[name] = {k: summary[k] for k in (
            "parameters", "global_step", "best_validation_nll", "best_validation_step", "training_exposure",
            "optimizer_skipped_steps", "applied_optimizer_updates")}
        results[name]["best_model"] = str(out/name/"best")
        results[name]["retained_initial_model"] = summary["continuation"]["retained_initial_model"]
        results[name]["initial_validation_nll"] = summary["continuation"]["initial_validation_nll"]
        atomic_json(out/"comparison.json", {"complete": len(results) == 2, "branches": results,
                    "benchmark_sha256": benchmark_hash, "test_evaluated": False, "matched_compute": False,
                    "matched_target_counts": False, "coherence_winner": None,
                    "interpretation": "Assess prose and copying against the unchanged baseline and trained control."})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    key = {}
    for selection_name in ("best", "final"):
        destinations = [("baseline", out/"control/initial", out/"starting_model")]
        for name in results:
            folder = out/name/("best" if selection_name == "best" else f"reviews/step-{args.max_steps}")
            destinations.append((name, folder, folder if selection_name == "best" else folder/"model"))
        for panel, field in (("full", "samples"), ("short", "short_samples")):
            rows = []
            for name, folder, checkpoint in destinations:
                for sample in json.loads((folder/"coherence.json").read_text())[field]:
                    opaque = hashlib.sha256(
                        f"{benchmark_hash}:{name}:{selection_name}:{panel}:{sample['sample_id']}".encode()).hexdigest()[:16]
                    key[opaque] = {"branch": name, "selection": selection_name, "panel": panel,
                                   "sample_id": sample["sample_id"], "checkpoint": str(checkpoint)}
                    rows.append({**sample, "sample_id": opaque})
            random.Random(args.seed).shuffle(rows)
            path = out/f"blind_{selection_name}_{panel}.csv"
            if not path.exists():  # Preserve ratings if a finished experiment is resumed.
                write_coherence_sheet(path, rows)
    atomic_json(out/"review_key.json", key)
    print(f"\nScene pilot finished. Read {out/'comparison.json'} and blind_best_full.csv / "
          "blind_final_full.csv; the short panels are prefixes of those same outputs. "
          "Best-NLL selection can retain the starting weights; every review snapshot remains exportable. "
          "Test evaluation was not run.", flush=True)
    return results


def select_winter_text(body, spec):
    """Fail closed when an edition changes; never guess a replacement boundary."""
    matches = list(re.finditer(spec["start"], body, re.M))
    if len(matches) != 1:
        raise ValueError(f"Winter start boundary must be unique: {spec['start']}")
    start = matches[0]
    end = re.search(spec["end"], body[start.end():], re.M)
    if end is None:
        raise ValueError(f"Missing winter end boundary: {spec['end']}")
    stop = start.end()+end.start()
    text = body[start.start():stop].strip()
    text = re.sub(r'(?:\n\s*(?:[IVX]+\.|(?:\*\s*){3,}))+\s*$', '', text).strip()
    if hashlib.sha256(text.encode()).hexdigest() != spec["sha256"]:
        raise ValueError("Selected Gutenberg text changed from the reviewed edition. "
                         "Restore the cached edition or review/update the inline selection; do not bypass this check.")
    if re.search(r'Project Gutenberg|Transcriber[’\']s [Nn]otes', text):
        raise ValueError("Boilerplate survived winter selection")
    return text, start.start(), stop


def word_shingles(words, width=32):
    for start in range(len(words)-width+1):
        key = hashlib.blake2b(" ".join(words[start:start+width]).encode(), digest_size=8).digest()
        yield start, key


def screen_winter_units(units, original):
    """Exclude shared held-out passages and substantial training reprints.

    `original` contains decoded SAVED token sequences, not mutable clean files.
    Every hash hit is verified for the 32-word held-out check. Shorter paraphrases
    and related-series similarities are not claimed to be detected.
    """
    heldout, index, training = [], defaultdict(list), []
    for row in original:
        if row["split"] == "train":
            training.append((row["id"], fingerprints(row["text"])))
        else:
            words = re.findall(r'\w+', row["text"].casefold())
            number = len(heldout)
            heldout.append((row, words))
            for start, key in word_shingles(words):
                index[key].append((number, start))
    accepted, excluded = [], []
    for row in units:
        words = re.findall(r'\w+', row["text"].casefold())
        conflict = None
        for start, key in word_shingles(words):
            for number, other_start in index.get(key, ()):
                other, other_words = heldout[number]
                if words[start:start+32] == other_words[other_start:other_start+32]:
                    conflict = {"reason": "heldout_exact_32_word_overlap", "other_id": other["id"],
                                "split": other["split"], "start_word": start, "other_start_word": other_start}
                    break
            if conflict:
                break
        sig = fingerprints(row["text"])
        if not conflict:
            for other_id, other_sig in training:
                shared = len(sig & other_sig)
                containment = shared/max(1, min(len(sig), len(other_sig)))
                if shared >= 12 and containment >= .05:
                    conflict = {"reason": "substantial_training_or_candidate_reprint", "other_id": other_id,
                                "shared_sampled_13_word_shingles": shared, "containment": containment}
                    break
        if conflict:
            excluded.append({"id": row["id"], **conflict})
        else:
            accepted.append(row)
            training.append((row["id"], sig))
    return accepted, excluded


def prepare_winter_material(data, cache, tokenizer, signature, offline=False):
    """Download the approved inline sources, select exact prose, and freeze training IDs."""
    units, records = [], []
    for source in WINTER_SOURCES:
        raw = download_book(source["url"], cache, offline)
        metadata, body = clean_book(raw)
        records.append({**source, **metadata, "raw_sha256": hashlib.sha256(raw.encode()).hexdigest()})
        for number, spec in enumerate(source["selections"], 1):
            text, start, end = select_winter_text(body, spec)
            units.append({"id": f"winter-{source['id']}-{number}", "source_id": source["id"],
                          "text": text, "sha256": spec["sha256"], "natural_end": spec["natural_end"],
                          "words": len(text.split()), "clean_start_character": start,
                          "clean_end_character_exclusive": end,
                          **{k: spec[k] for k in ("title", "year", "date_evidence", "fit", "caveats") if k in spec}})
        print(f"[winter source {len(records)}/{len(WINTER_SOURCES)}] {metadata['title']}", flush=True)
    original = []
    for split in ("train", "validation", "test"):
        for line in (Path(data)/f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            original.append({"id": str(row["id"]), "split": split,
                             "text": tokenizer.decode(row["ids"], skip_special_tokens=True)})
    accepted, excluded = screen_winter_units(units, original)
    if not accepted:
        raise ValueError("No additional material passed the overlap checks")
    lines, selection = [], []
    for row in accepted:
        ids = [tokenizer.eos_token_id]+tokenizer.encode(row["text"], add_special_tokens=False)
        if row["natural_end"]:
            ids.append(tokenizer.eos_token_id)
        lines.append(json.dumps({"id": row["id"], "ids": ids}))
        selection.append({k: v for k, v in row.items() if k != "text"} | {"target_tokens": len(ids)-1})
    encoded = "\n".join(lines)+"\n"
    manifest = {"version": 2, "catalog_sha256": json_digest(WINTER_SOURCES),
                "base_dataset_sha256": signature, "train_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "sources": records, "accepted": selection, "excluded": excluded,
                "words": sum(r["words"] for r in selection),
                "target_tokens": sum(r["target_tokens"] for r in selection),
                "test_evaluated": False,
                "screen": "Held-out texts read only for exact 32-normalized-word overlap exclusion; "
                          "no model evaluation. Train/new reprints screened with sampled 13-word fingerprints, "
                          ">=12 shared and >=5% smaller-document containment. Not a semantic deduplication guarantee."}
    return manifest, encoded


def combined_book_windows(first, second):
    """Unweighted union for copy detection and fixed-subset diagnostics."""
    import copy
    result = copy.copy(first)
    result.books = first.books+second.books
    result.book_ids = first.book_ids+second.book_ids
    result.windows = first.windows+[(b+len(first.books), start) for b, start in second.windows]
    result.random_offsets = False
    result.original_pool, result.supplementary_pool = first, second
    result.copy_reference_scope = ("Original training plus additional winter selections for ALL comparisons, "
                                   "including the unchanged baseline and optional control. A match to an addition "
                                   "does not mean the baseline/control trained on it.")
    result.set_training_epoch(0)
    return result


class WinterReplayDataset:
    """Half new / half original windows, with all original windows once per epoch.

    New windows cycle without replacement; each cycle gets fresh book offsets.
    The context look-back has no loss labels. Tail lengths cause small differences
    from a 50% target share, reported exactly by exposure().
    """
    def __init__(self, replay, supplementary, seed=42):
        if replay.block_size != supplementary.block_size:
            raise ValueError("Winter and replay windows must have the same context length")
        self.replay, self.supplementary, self.seed = replay, supplementary, seed
        self.block_size, self.random_offsets = replay.block_size, True
        self.set_training_epoch(0)

    def __len__(self):
        return 2*len(self.replay)

    def set_training_epoch(self, epoch):
        import copy
        self.epoch = int(epoch)
        self.replay.set_training_epoch(self.epoch)
        self.cycles, self.order = [], []
        cycles = math.ceil(len(self.replay)/len(self.supplementary))
        for cycle in range(cycles):
            dataset = copy.copy(self.supplementary)
            dataset.random_offsets = True
            dataset.set_training_epoch(self.epoch*cycles+cycle)
            self.cycles.append(dataset)
            order = list(range(len(dataset)))
            random.Random(f"winter:{self.seed}:{self.epoch}:{cycle}").shuffle(order)
            self.order.extend((cycle, i) for i in order)
        self.order = self.order[:len(self.replay)]

    def __getitem__(self, index):
        slot, replay = divmod(index, 2)
        if replay:
            return self.replay[slot]
        cycle, row = self.order[slot]
        return self.cycles[cycle][row]

    def exposure(self, effective_batch, steps, seed):
        import torch
        previous = self.epoch
        per_epoch = math.ceil(len(self)/effective_batch)
        result = {"steps": steps, "updates_per_epoch": per_epoch, "complete_epochs": steps//per_epoch,
                  "examples": 0, "scored_targets": 0, "new_examples": 0, "new_targets": 0,
                  "replay_examples": 0, "replay_targets": 0}
        remaining = steps
        try:
            for epoch in range(math.ceil(steps/per_epoch)):
                self.set_training_epoch(epoch)
                order = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed+epoch)).tolist()
                for index in order[:min(remaining, per_epoch)*effective_batch]:
                    row = self[index]
                    targets = sum(x != -100 for x in row.get("labels", row["input_ids"])[1:])
                    kind = "replay" if index % 2 else "new"
                    result[kind+"_examples"] += 1
                    result[kind+"_targets"] += targets
                    result["examples"] += 1
                    result["scored_targets"] += targets
                remaining -= min(remaining, per_epoch)
        finally:
            self.set_training_epoch(previous)
        unique_targets = sum(len(book)-1 for book in self.supplementary.books)
        result["new_fraction_of_scored_targets"] = result["new_targets"]/max(1, result["scored_targets"])
        result["new_corpus_target_pass_equivalents"] = result["new_targets"]/unique_targets
        return result


def default_winter_source(root):
    latest = Path(root)/"runs/christmas-winter/expanded/best"
    return latest if latest.is_dir() else Path(root)/"runs/christmas-scenes/control/best"


def coherence_settings(args):
    config = {'version': 1, 'positions': args.coherence_context,
            'curriculum_fraction': args.curriculum_fraction,
            'context_weight': args.context_loss_weight, 'context_margin': args.context_margin,
            'unlikelihood_weight': args.unlikelihood_weight, 'unlikelihood_every': args.unlikelihood_every,
            'unlikelihood_tokens': args.unlikelihood_tokens, 'rerank_candidates': args.rerank_candidates}
    if args.command == 'train-continuity':
        config.update(version=2, continuity_weight=args.continuity_weight,
            continuity_positive_weight=.5, continuity_beta=2., continuity_margin=.2,
            continuity_pairs_per_batch=2, continuity_pair_version=1,
            online_weight=args.online_preference_weight, online_every=2, online_tokens=64, online_beta=2.)
    return config


def finetune_winter(args):
    """Continue project weights with an explicit, independently frozen expansion."""
    import gc
    import os
    import torch
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, set_seed
    continuity = args.command == 'train-continuity'
    coherence = coherence_settings(args) if args.command in {'train-coherence', 'train-continuity'} else None

    if args.corpus or args.rebuild or args.grow_layers or args.continuation_dropout is not None:
        raise ValueError("Winter/coherence continuation fixes its source catalog, tokenizer, blocks and inherited dropout")
    if args.resume not in {None, "latest"}:
        raise ValueError("Use --resume without a checkpoint path for the winter experiment")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
            torch.cuda.is_available() and not args.cpu and torch.cuda.device_count() != 1):
        raise ValueError("Expose one training device/process for reproducible winter replay")
    data, out = Path(args.data_dir), Path(args.output_dir)
    if not (data/"report.json").is_file():
        raise ValueError("Restore the source model's prepared data/ directory. This continuation never rebuilds it.")
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Use a new winter output directory or --resume, including after --plan-only")
    if args.resume and not (out/"experiment.json").is_file():
        raise ValueError("No frozen winter experiment to resume")
    if args.max_steps <= 0 or args.max_steps < args.eval_steps:
        raise ValueError("Winter budget must be positive and at least --eval-steps")
    if args.train_limit or args.eval_limit:
        raise ValueError("Use --smoke-test for execution checks; full runs keep all prepared data")
    prepare(args)
    signature = dataset_signature(data)
    tokenizer = AutoTokenizer.from_pretrained(data/"tokenizer", local_files_only=True)
    latest = ROOT/'runs/christmas-winter-v2/expanded/best'
    default_source = latest if coherence and latest.is_dir() else default_winter_source(ROOT)
    if continuity:
        for candidate in (ROOT/'runs/christmas-coherence/expanded/best', ROOT/'runs/christmas-coherence/expanded/last'):
            if candidate.is_dir():
                default_source = candidate
                break
    source = Path(args.continue_from or default_source).resolve()
    snapshot, material = out/"starting_model", out/"material"
    if out.resolve() == source or out.resolve() in source.parents:
        raise ValueError("Winter output must not contain or replace its source checkpoint")
    if args.resume:
        prior = json.loads((out/"experiment.json").read_text())
        model = load_continuation(snapshot, tokenizer, signature)
        if continuity and coherence['positions'] is None:
            coherence['positions'] = model.config.n_positions
        source_description = prior["source"]
        manifest = json.loads((material/"supplementary_sources.json").read_text())
        encoded = (material/"train.jsonl").read_text()
        if (manifest["base_dataset_sha256"] != signature
                or manifest["catalog_sha256"] != json_digest(WINTER_SOURCES)
                or manifest["train_sha256"] != hashlib.sha256(encoded.encode()).hexdigest()):
            raise ValueError("Frozen winter material, catalog or base data changed. Start v2 in a NEW output "
                             "directory with --continue-from pointing to the previous selected weights; "
                             "an old optimizer run cannot be resumed with a different corpus/recipe.")
    else:
        if args.smoke_test:
            set_seed(args.model_seed)
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=len(tokenizer), n_positions=512, n_ctx=512, n_layer=1, n_embd=32,
                n_head=2, n_inner=128, bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                resid_pdrop=.05, embd_pdrop=.05, attn_pdrop=.05))
            source_description = {"kind": "random_tiny_execution_check", "seed": args.model_seed}
        else:
            if not source.is_dir():
                raise ValueError(f"Source weights not found: {source}. Set --continue-from to an existing "
                                 "project best/ directory (the latest winter best or scenes/control/best). "
                                 "Reports ZIPs do not contain weights.")
            model = load_continuation(source, tokenizer, signature)
            print(f"Winter continuation source: {source}", flush=True)
            source_description = {"kind": "existing_project_model", "path": str(source),
                                  "parameters": model.num_parameters()}
            provenance = next((source/name for name in ("training_summary.json", "provenance.json")
                               if (source/name).exists()), None)
            if provenance:
                source_description["parent_provenance"] = json.loads(provenance.read_text())
        if not coherence and model.config.n_positions != 512:
            raise ValueError("The reviewed winter experiment requires a context-512 project checkpoint")
        if coherence:
            previous = model
            if coherence['positions'] is None:
                coherence['positions'] = previous.config.n_positions
            model, extension = extend_gpt2_context(previous, coherence['positions'])
            if extension['changed']:
                extension['verification'] = verify_depth_growth(previous, model, tokenizer)
            source_description['context_extension'] = extension
            model.config.coherence_decoding = {'candidates': coherence['rerank_candidates']}
            del previous
        manifest, encoded = prepare_winter_material(
            data, Path(args.winter_cache or data.parent/"winter-data"), tokenizer, signature, args.offline)
    args.block_size = model.config.n_positions
    if coherence and not getattr(args,'winter_lookback_explicit',True):
        args.winter_lookback=min(args.block_size//4,256)
    pair_records = None
    validation_pairs = None
    if continuity:
        if args.resume:
            pair_records = json.loads((material/'continuity_train.json').read_text())
            validation_pairs = json.loads((material/'continuity_validation.json').read_text())
        else:
            # Build directly from token arrays; do not touch the held-out test books.
            original = BookWindows(data/'train.jsonl', 512)
            extra_rows = [json.loads(line) for line in encoded.splitlines()]
            import numpy as np
            from types import SimpleNamespace
            extra = SimpleNamespace(book_ids=[str(row['id']) for row in extra_rows],
                                    books=[np.asarray(row['ids'], dtype=np.int32) for row in extra_rows])
            pair_records = build_continuity_pairs(original, tokenizer, seed=args.seed)
            pair_records += build_continuity_pairs(extra, tokenizer, seed=args.seed)
            validation_pairs = build_continuity_pairs(BookWindows(data/'validation.jsonl', 512), tokenizer,
                                                       per_book=12, seed=args.seed+1)
        if not pair_records or not validation_pairs:
            raise ValueError('No sentence-bounded continuity pairs; inspect prepared text/tokenizer')
        coherence['training_pairs_sha256'] = json_digest(pair_records)
        coherence['validation_pairs_sha256'] = json_digest(validation_pairs)
        print(f'Continuity supervision: {len(pair_records):,} training pairs, '
              f'{len(validation_pairs):,} validation-only pairs.', flush=True)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}".encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if args.smoke_test:
        args.cpu, args.precision = True, "fp32"
        args.batch_size, args.accumulation, args.max_steps = 2, 1, 3
        args.train_limit, args.eval_limit = 16, 8
        args.eval_steps, args.sample_steps, args.benchmark_steps = 2, 2, 2
        args.sample_tokens, args.benchmark_tokens, args.benchmark_prompts = 12, 12, 2
    args.precision = resolve_precision(args.precision, args.cpu)
    benchmark = make_capacity_benchmark(data, tokenizer, signature, args.benchmark_prompts,
                                       args.seed, generation_tokens=args.benchmark_tokens)
    benchmark["short_tokens"] = min(96, args.benchmark_tokens)
    benchmark["context_probe_tokens"] = min(64, args.benchmark_tokens)
    benchmark["context_probe_version"] = 1
    if coherence:
        benchmark['reranked_prompts'] = 1 if args.smoke_test else min(3, args.benchmark_prompts)
    if continuity:
        benchmark['continuity_pairs'] = validation_pairs[:4] if args.smoke_test else validation_pairs
    if args.smoke_test:
        benchmark["training_eval_windows"] = 16
    recipe = {"version": 2, "manifest_sha256": json_digest(manifest),
              "train_sha256": manifest["train_sha256"], "new_example_fraction": .5,
              "lookback": args.winter_lookback,
              "prefix_loss": "preceding_context_only_masked; all_target_span_tokens_scored", "false_end_targets": False}
    if coherence:
        recipe['coherence'] = coherence
    settings = {k: getattr(args, k) for k in (
        "seed", "model_seed", "batch_size", "accumulation", "learning_rate", "warmup_ratio", "max_steps",
        "eval_steps", "sample_steps", "sample_tokens", "benchmark_steps", "precision", "smoke_test",
        "train_limit", "eval_limit", "with_control", "winter_lookback")}
    plan = {"version": 2, "source": source_description, "starting_weights_sha256": digest.hexdigest(),
            "dataset_sha256": signature, "recipe": recipe, "settings": settings,
            "benchmark_sha256": json_digest(benchmark), "test_evaluated": False,
            "rationale": {
                "material": f"All {len(WINTER_SOURCES)} shortlisted sources processed. Exact complete stories or contiguous "
                "winter/domestic chapter ranges; no rewriting, keyword-filtered fragments or external weights.",
                "replay": "50% additional / 50% original examples; both pools have the same target-span width "
                "and look-back. Every original window once per mixture epoch. New windows cycle without replacement "
                "with fresh offsets per cycle. Exact target counts and new-corpus pass equivalents are reported.",
                "endings": "No windows cross selected ranges. EOS only at genuine story/book ends; chapter cuts "
                "are not declared complete stories. Original spellings, dialogue and verse are retained.",
                "context": "Prepend up to --winter-lookback prior tokens within the same document; mask only "
                "this added context and the first conditioning token. At the default 128, full rows score 383 "
                "targets in 512 positions. Use 0 for a no-look-back ablation. Shifts still sample interior targets; "
                "padding/context labels never enter target accounting. Validation/review loss windows unchanged.",
                "optimization": "Inherited GPT-2 architecture/dropout/tokenizer. Fresh AdamW, default peak 5e-5, "
                "betas .9/.95, decay .1, clip 1; 3% warmup, cosine to 5e-6. Default batch 8x4, 1000 updates. "
                "Conservative pilot settings, not tuned optima; new material remains small.",
                "evaluation": "Original validation and generation prompts fixed; test read only for overlap screening. "
                "Baseline eligible for best NLL; all review weights retained. All new selections are training "
                "material, not a new validation split. Reference NLL with real versus unrelated, equal-length "
                "context measures context sensitivity, not coherence. No automatic coherence verdict.",
                "control": "Optional same-initialization, same-update ordinary continuation. Target counts and "
                "compute are not exactly matched; compare prose as well as losses."}}
    if coherence:
        plan['rationale'].update(
            context='Extend positional embeddings and causal buffers to the configured length, copying old '
                    'positions exactly. Added positions are tiled initialization requiring training. Validation '
                    'and dropout-off review NLL still use 512-token windows.',
            curriculum='Anneal bounded passage-simplicity weights to one over the first configured fraction '
                       'of updates. Normalize within each pool per microbatch; no books or positive targets removed.',
            context_objective='Weight 0.1 by default: hinge margin 0.2 nats/token, true vs different-book prefix '
                              'for identical 64-token reference; 128-token contexts, dropout off, original CE retained.',
            unlikelihood='Weight 0.1 by default: once every 8 optimizer updates, greedily generate up to 64 tokens '
                         'from one training prefix; penalize only third+ occurrences of 8-token phrases. '
                         'No generated text becomes a positive target. Actual negative counts are logged.',
            decoding='Sample eight candidates, score fluency plus bounded full-vs-short-context gain, '
                     'minus phrase repetition and prompt copying. Preserve raw prose and all candidate scores. '
                     'This is a same-model heuristic, not an independent coherence judge.',
            optimization='Single combined continuation; defaults batch 4 x accumulation 4, peak LR 3e-5, '
                         '1000 updates, 3% warmup and cosine to 3e-6. Inherit dropout and transformer blocks. '
                         'Longer positive spans approximately preserve the previous positive-target update budget. '
                         'Auxiliary forwards/greedy mining add compute. Settings are hypotheses, not measured optima.',
            control='Optional original-corpus-only branch; all five methods also apply to that branch. '
                    'It isolates supplementary material, not the new objectives; budgets are not compute-matched.')
    if continuity:
        plan['rationale'].update(
            continuity='Original sentence-bounded continuations are positive. Negatives alter a name, auxiliary '
                       'polarity, or sentence order in that same passage. These are automatic corruptions, '
                       'not human correctness labels. Pairwise softplus on up to 32 tokens after the first '
                       'difference, beta 2, gap margin .2, weight .5; positive continuation CE weight .5. '
                       'All source replay CE remains. At most two pairs per microbatch. Validation books '
                       'produce evaluation pairs only; no test-book preference examples.',
            online_preferences='Every two optimizer updates, generate up to 64 tokens from one training prefix, '
                               'alternating greedy and sampled rollouts. Prefer the actual source continuation '
                               'using length-normalized DPO-style log ratios against a frozen copy of this '
                               'project starting model (beta 2, weight .5). Generated prose is never positive. '
                               'A different continuation is not automatically incorrect; this is a preference '
                               'for the observed source. This objective uses no outside model or annotations. '
                               'Mining preserves RNG streams and runs once per scheduled optimizer update.',
            context='Inherit the source context unless explicitly overridden. Keep validation windows at 512.',
            optimization='Default 1500 updates, LR 2e-5, batch 4 x accumulation 4, inherited dropout, '
                         '3% warmup and cosine decay. Disable simplicity weighting and other-book prefix '
                         'ranking by default; concentrate auxiliary supervision on local continuation choices.')
    if args.resume:
        if prior != plan or json.loads((out/"benchmark.json").read_text()) != benchmark:
            raise ValueError("Frozen winter settings, source weights, material or benchmark changed")
    else:
        material.mkdir(parents=True, exist_ok=True)
        (material/"train.jsonl").write_text(encoded)
        if continuity:
            atomic_json(material/'continuity_train.json', pair_records)
            atomic_json(material/'continuity_validation.json', validation_pairs)
        atomic_json(material/"supplementary_sources.json", manifest)
        model.save_pretrained(snapshot, safe_serialization=True)
        tokenizer.save_pretrained(snapshot)
        atomic_json(snapshot/"provenance.json", {"dataset_sha256": signature, "source": source_description})
        atomic_json(out/"benchmark.json", benchmark)
        atomic_json(out/"experiment.json", plan)
    replay = ContextBookWindows(data/"train.jsonl", args.block_size, args.train_limit, random_offsets=True,
                                seed=args.seed, lookback=args.winter_lookback)
    new = ContextBookWindows(material/"train.jsonl", args.block_size, args.train_limit, random_offsets=True,
                             seed=args.seed, lookback=args.winter_lookback)
    exposure = training_exposure(WinterReplayDataset(replay, new, args.seed),
                                args.batch_size*args.accumulation, args.max_steps, args.seed)
    atomic_json(out/"planned_exposure.json", exposure)
    print("\nWINTER CONTINUATION:", json.dumps({"sources_processed": len(manifest["sources"]),
          "accepted_ranges": len(manifest["accepted"]), "excluded": manifest["excluded"],
          "additional_words": manifest["words"], "additional_targets": manifest["target_tokens"],
          "planned_exposure": exposure, "settings": settings}, indent=2), flush=True)
    del model, replay, new
    gc.collect()
    if args.plan_only:
        print("Plan and source snapshot saved. Add --resume to train the unchanged plan.", flush=True)
        return plan
    results = {}
    branches = ("control", "expanded") if args.with_control else ("expanded",)
    for name in branches:
        branch = argparse.Namespace(**vars(args))
        branch.output_dir, branch.command = str(out/name), "train"
        branch.continue_from = str(snapshot.resolve())
        branch.fixed_budget, branch.skip_test, branch.random_window_offsets = True, True, True
        branch.capacity_benchmark, branch.benchmark_sha256 = benchmark, json_digest(benchmark)
        branch.winter_recipe = {**recipe, "treatment": name == "expanded"}
        branch.winter_train_path, branch.winter_manifest = str(material/"train.jsonl"), manifest
        if continuity:
            allowed = {json.loads(line)['id'] for line in (data/'train.jsonl').read_text().splitlines()}
            branch.continuity_pairs = pair_records if name == 'expanded' else [p for p in pair_records if p['book_id'] in allowed]
        branch.resume = "latest" if args.resume and (out/name/"run_config.json").exists() else None
        if branch.resume and latest_complete_checkpoint(out/name, require_scaler=args.precision == "fp16"):
            branch.continue_from = None
        complete = out/name/"best/training_summary.json"
        if branch.resume and complete.exists() and (out/name/"best/coherence.json").exists():
            summary = json.loads(complete.read_text())
            if (summary["global_step"] != args.max_steps or summary["dataset_sha256"] != signature
                    or summary["run"].get("winter_recipe") != branch.winter_recipe):
                raise ValueError(f"Completed winter branch differs from its frozen plan: {name}")
        else:
            summary = training(branch)
        results[name] = {k: summary[k] for k in (
            "parameters", "global_step", "best_validation_nll", "best_validation_step", "training_exposure",
            "optimizer_skipped_steps", "applied_optimizer_updates")}
        results[name].update(best_model=str(out/name/"best"), **{k: summary["continuation"][k] for k in (
            "retained_initial_model", "initial_validation_nll")})
        if coherence:
            results[name]['last_model'] = str(out/name/'last')
        results[name]["context_use"] = {}
        for label, folder in (("initial", out/name/"initial"), ("selected", out/name/"best")):
            context_use = json.loads((folder/"coherence.json").read_text()).get("context_use")
            if context_use:
                results[name]["context_use"][label] = {k: context_use[k] for k in (
                    "paired_prompts", "scored_tokens_per_condition", "true_prefix_nll",
                    "unrelated_prefix_nll", "context_gain_nll")}
        atomic_json(out/"comparison.json", {"complete": len(results) == len(branches), "branches": results,
                    "benchmark_sha256": json_digest(benchmark), "test_evaluated": False,
                    "matched_compute": False, "matched_target_counts": False, "coherence_winner": None,
                    "interpretation": "Compare prose against the unchanged baseline; NLL is not coherence."})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    key = {}
    for selection in ("best", "final"):
        destinations = [("baseline", out/branches[0]/"initial", snapshot)]
        for name in branches:
            folder = out/name/("best" if selection == "best" else f"reviews/step-{args.max_steps}")
            destinations.append((name, folder, folder if selection == "best" else folder/"model"))
        for panel, field in (("full", "samples"), ("short", "short_samples")):
            rows = []
            for name, folder, checkpoint in destinations:
                for sample in json.loads((folder/"coherence.json").read_text())[field]:
                    opaque = hashlib.sha256(
                        f"{plan['benchmark_sha256']}:{name}:{selection}:{panel}:{sample['sample_id']}".encode()).hexdigest()[:16]
                    key[opaque] = {"branch": name, "selection": selection, "panel": panel,
                                   "sample_id": sample["sample_id"], "checkpoint": str(checkpoint)}
                    rows.append({**sample, "sample_id": opaque})
            random.Random(args.seed).shuffle(rows)
            path = out/f"blind_{selection}_{panel}.csv"
            if not path.exists():
                write_coherence_sheet(path, rows)
    atomic_json(out/"review_key.json", key)
    if coherence:
        import shutil
        # Preserve the trained outcome even when the auxiliary objectives worsen NLL.
        # best/ remains explicitly NLL-selected; last/ is explicitly the final update.
        for name in branches:
            shutil.copytree(out/name/f'reviews/step-{args.max_steps}/model', out/name/'last', dirs_exist_ok=True)
            (out/name/'last/README.md').write_text(
                '# Christmasmodel: final coherence continuation\n\n'
                f'Final optimizer step: {args.max_steps}. This is not a coherence-selected winner. '
                'Original Christmas stories and approved winter selections; project weights and tokenizer only. '
                'The base model was randomly initialized. See training_summary.json and supplementary_sources.json.\n\n'
                'The training_summary.json records the active auxiliary objectives and context length. '
                'Candidate reranking is inference code in model.py; GGUF contains weights only. '
                'Other inference engines do not automatically apply this reranker.\n', encoding='utf-8')
        print(f"Final trained weights: {out/'expanded/last'}; best/ remains selected by validation NLL.", flush=True)
    print(f"Winter continuation finished: {out/'comparison.json'}. Review blind_best_full.csv and "
          "blind_final_full.csv; every periodic review retains exportable weights. Test not evaluated.", flush=True)
    return results


def model_card(summary):
    return f"""---
language: en
library_name: transformers
pipeline_tag: text-generation
tags:
- gpt2
- from-scratch
- christmas
---
# Christmasmodel

A {summary['parameters']:,}-parameter GPT-2 language model trained on the Gutenberg works listed in
[christmasmodel](https://github.com/wildfellhall/christmasmodel/blob/main/corpus.py).
{('This run used the selected pre-1920 winter/domestic sources documented in supplementary_sources.json. The original validation/test splits and tokenizer remain fixed. If retained_initial_model is true below, this selected export retains the starting weights.' if (summary['run'].get('winter_recipe') or {}).get('treatment') else '')}
Its byte-level BPE vocabulary was learned on training books only.
The base model starts from random weights. Continuations reuse only this project's
own weights; optional depth expansion inserts initially identity residual blocks.

This is a literary continuation model, not an instruction/chat model. Use a
story opening as the prompt. Coherence, novelty, and factual accuracy are not
guaranteed. The limited historical corpus can produce memorized passages,
repetition, dated language, and historical biases. Repeated epochs are not new data.

Best validation NLL: {summary['best_validation_nll']}
Continuation details: {json.dumps(summary.get('continuation'))}
Training windows: {'shifted within books each epoch' if summary['run'].get('random_window_offsets') else 'fixed'}
Run intervention: {'Winter-fiction additions with 50% original replay examples; the saved winter_recipe specifies the context look-back and target supervision.' if (summary['run'].get('winter_recipe') or {}).get('treatment') else ('Original-text scene continuations with masked prefixes and corpus replay.' if (summary['run'].get('scene_recipe') or {}).get('treatment') else 'Ordinary causal language modeling.')}
Test token perplexity: {summary['test'].get('test_perplexity', 'not evaluated in this development run')}
Smoke test: {summary['smoke_test']}; limited data: {summary['limited_data']}.
Scores use this model's tokenizer and are not directly comparable to other vocabularies.
Overlap-group splits reduce, but cannot guarantee the absence of, shared passages.
Full corpus counts, settings, and evaluation are in training_summary.json.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
path = "./runs/christmas/best"  # or the uploaded Hugging Face repository ID
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)
inputs = tokenizer("On Christmas Eve, the snow", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=96, do_sample=True,
                        temperature=0.8, top_p=0.92, top_k=40,
                        repetition_penalty=1.08,
                        pad_token_id=tokenizer.pad_token_id,
                        suppress_tokens=[tokenizer.pad_token_id])
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
"""


def corpus_probe_inputs(dataset, tokenizer, max_new_tokens, seed, count=3):
    """Choose deterministic middle-of-book passages with their original token IDs."""
    rng = random.Random(seed)
    probes = []
    for book in rng.sample(range(len(dataset.books)), min(count, len(dataset.books))):
        tokens = dataset.books[book]
        start = max(1, len(tokens) // 2)
        window = tokens[start:start+dataset.block_size].tolist()
        prefix_length = min(64, dataset.block_size // 3, len(window)-2)
        if prefix_length < 1:
            continue
        ids = window[:prefix_length]
        reference = window[prefix_length:prefix_length+max_new_tokens]
        probes.append({"book_id": dataset.book_ids[book], "start_token": start,
                       "input_ids": ids, "prefix_tokens": prefix_length,
                       "prompt": tokenizer.decode(ids, skip_special_tokens=True),
                       "reference_ids": reference,
                       "reference_continuation": tokenizer.decode(reference, skip_special_tokens=True)})
    return probes


def diagnose(args):
    """Compare train/validation in inference mode, then probe known book passages."""
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    if not args.model_dir:
        raise ValueError("diagnose requires --model-dir pointing to a saved project model")
    path = Path(args.model_dir)
    data, out = Path(args.data_dir), Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new --output-dir for diagnostics; the training run is left untouched")
    signature = dataset_signature(data)
    manifest = json.loads((data / "manifest.json").read_text())
    if any(book["url"] not in sources for book in manifest):
        raise ValueError("Diagnostic data include sources outside the approved Christmas corpus")
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    # Reuse the project's provenance, tokenizer, and special-ID checks without
    # creating an optimizer, a training stage, or a new checkpoint.
    model = load_continuation(path, tokenizer, signature)
    model.config.use_cache = False
    model.eval()
    datasets = {split: BookWindows(data / f"{split}.jsonl", model.config.n_positions, args.diagnostic_windows)
                for split in ("train", "validation")}
    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir=str(out), per_device_eval_batch_size=args.batch_size,
        use_cpu=args.cpu, report_to="none", dataloader_num_workers=0),
        data_collator=CausalCollator(tokenizer.pad_token_id),
        preprocess_logits_for_metrics=loss_sums, compute_metrics=token_metrics)
    metrics = {}
    for split, dataset in datasets.items():
        prefix = f"{split}_eval"
        result = trainer.evaluate(dataset, metric_key_prefix=prefix)
        metrics[split] = {"nll": result[f"{prefix}_nll"], "perplexity": result[f"{prefix}_perplexity"],
                          "scored_tokens": result[f"{prefix}_scored_tokens"], "windows": len(dataset)}
    gap = metrics["validation"]["nll"] - metrics["train"]["nll"]
    probes = {}
    for split, dataset in datasets.items():
        probes[split] = []
        for probe in corpus_probe_inputs(dataset, tokenizer, args.sample_tokens, args.seed):
            generated = generate_samples(model, tokenizer, [probe], args.sample_tokens, args.seed)
            record = {**probe, "generated": generated}
            probes[split].append(record)
            print(f"\n[{split} book {probe['book_id']}] PREFIX: {probe['prompt']}\n"
                  f"REFERENCE: {probe['reference_continuation']}", flush=True)
            for sample in generated:
                print(f"{sample['mode'].upper()}: {sample['continuation']}\n", flush=True)
    report = {"model_dir": str(path.resolve()), "dataset_sha256": signature,
              "dropout_disabled": True, "weights_updated": False, "test_evaluated": False,
              "window_limit_per_split": args.diagnostic_windows, "metrics": metrics,
              "validation_minus_training_nll": gap,
              "weight_tying_intact": model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr(),
              "probes": probes,
              "interpretation": "Both losses use inference mode and token weighting. Training windows are "
              "a deterministic subset when capped. Whole-book splits can include style differences; "
              "there is no universal NLL-gap threshold for overfitting. Greedy loops alone do not diagnose "
              "capacity, optimization, or data limitations. Passage probes are qualitative examples, "
              "not a numerical coherence score."}
    write_json(out / "diagnostics.json", report)
    print("\nDIAGNOSTIC SUMMARY (dropout off; no training or test-set evaluation):")
    print(json.dumps({"metrics": metrics, "validation_minus_training_nll": gap,
                      "weight_tying_intact": report["weight_tying_intact"]}, indent=2))
    print(f"Full report and passage probes: {out / 'diagnostics.json'}")
    return report


def repetition_counts(tokens, n=4):
    """Duplicate token n-grams within a continuation, not a coherence metric."""
    grams = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]
    return {"ngrams": len(grams), "repeated_ngrams": len(grams)-len(set(grams))}


def compare_decoding(args):
    """Bounded generation comparison on the same model, prefixes, and lengths."""
    import torch
    from transformers import AutoTokenizer
    if not args.model_dir:
        raise ValueError("compare-decoding requires --model-dir pointing to a saved project model")
    data, out = Path(args.data_dir), Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new --output-dir for the decoding comparison")
    signature = dataset_signature(data)
    manifest = json.loads((data / "manifest.json").read_text())
    if any(book["url"] not in sources for book in manifest):
        raise ValueError("Data include sources outside the approved Christmas corpus")
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    model = load_continuation(args.model_dir, tokenizer, signature)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else
                                     "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    prompts = [{"split": "fixed", "prompt": prompt,
                "input_ids": tokenizer.encode(prompt, add_special_tokens=False)} for prompt in PROMPTS]
    for split in ("train", "validation"):
        dataset = BookWindows(data / f"{split}.jsonl", model.config.n_positions, limit=1)
        prompts.extend({"split": split, **probe} for probe in
                       corpus_probe_inputs(dataset, tokenizer, args.sample_tokens, args.seed))
    modes = {
        "greedy": {"do_sample": False},
        "sample": {"temperature": .8, "top_p": .92, "top_k": 40, "repetition_penalty": 1.08},
        "blocked_sample": {"temperature": .8, "top_p": .92, "top_k": 40,
                           "repetition_penalty": 1.08, "no_repeat_ngram_size": 4},
        "contrastive": {"top_k": 4, "penalty_alpha": .6},
    }
    seeds = [args.seed, args.seed+1]
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for index, prompt in enumerate(prompts):
        print(f"\n[{index+1}/{len(prompts)} | {prompt['split']}] PREFIX: {prompt['prompt']}", flush=True)
        if "reference_continuation" in prompt:
            print(f"REFERENCE: {prompt['reference_continuation']}", flush=True)
        for mode in modes:
            for seed in (seeds if mode in {"sample", "blocked_sample"} else [args.seed]):
                # Start each method from the same prefix and RNG seed. Never
                # feed the reference or another method's generated tokens.
                sample, = generate_samples(model, tokenizer, [prompt], args.sample_tokens, seed,
                                           modes=(mode,), include_token_ids=True)
                ids = [token for token in sample["token_ids"]
                       if token not in {tokenizer.eos_token_id, tokenizer.pad_token_id}]
                record = {"prompt_index": index, "split": prompt["split"], "seed": seed, **sample,
                          "generated_tokens": len(ids),
                          "ended_with_eos": bool(sample["token_ids"] and
                                                 sample["token_ids"][-1] == tokenizer.eos_token_id),
                          **repetition_counts(ids)}
                results.append(record)
                with (out / "samples.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{mode.upper()} [seed {seed}]: {sample['continuation']}\n", flush=True)
    metrics = {}
    for mode in modes:
        rows = [row for row in results if row["mode"] == mode]
        total = sum(row["ngrams"] for row in rows)
        metrics[mode] = {
            "continuations": len(rows),
            "mean_generated_tokens": sum(row["generated_tokens"] for row in rows)/len(rows),
            "eos_endings": sum(row["ended_with_eos"] for row in rows),
            "repeated_token_4gram_fraction": sum(row["repeated_ngrams"] for row in rows)/total if total else None,
        }
    report = {"model_dir": str(Path(args.model_dir).resolve()), "dataset_sha256": signature,
              "weights_updated": False, "test_evaluated": False, "modes": modes, "sampling_seeds": seeds,
              "max_new_tokens": args.sample_tokens, "prompts": prompts, "metrics": metrics, "samples": results,
              "interpretation": "Duplicate 4-grams measure repetition, not meaning. Blocking forces this "
              "measure down and may impair prose. Compare lengths/EOS and read all outputs for scene, "
              "character, and grammatical consistency; no automatic winner is selected. A plausible "
              "continuation need not match the reference. These few passages cannot establish general quality."}
    write_json(out / "comparison.json", report)
    print("\nDECODING COMPARISON (same weights; repetition is not coherence):")
    print(json.dumps(metrics, indent=2))
    print(f"Full comparison: {out / 'comparison.json'}")
    return report


def inference(args):
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
    path = Path(args.model_dir or Path(args.output_dir) / "best")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(path, local_files_only=True)
    if args.command == "generate":
        device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else
                                        "mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        if args.rerank_candidates is not None:
            model.config.coherence_decoding = {'candidates': args.rerank_candidates}
        modes = ("greedy", "sample") if args.decode_mode == "both" else (args.decode_mode,)
        if args.decode_mode == 'both' and getattr(model.config, 'coherence_decoding', None):
            modes += ('reranked',)
        print(f'Model: {path.resolve()} ({model.num_parameters():,} parameters)', flush=True)
        for sample in generate_samples(model, tokenizer, [args.prompt], args.sample_tokens, args.seed, modes=modes):
            print_sample(sample)
        return
    summary = json.loads((path / "training_summary.json").read_text())
    if summary["dataset_sha256"] != dataset_signature(args.data_dir):
        raise ValueError("Evaluation data/tokenizer differ from this model's training run")
    coherence = (summary.get('winter_recipe') or summary.get('run', {}).get('winter_recipe') or {}).get('coherence')
    dataset = BookWindows(Path(args.data_dir) / "test.jsonl", 512 if coherence else model.config.n_positions, args.eval_limit)
    trainer = Trainer(model=model,
                      args=TrainingArguments(output_dir=str(path.parent / "evaluation"),
                                             per_device_eval_batch_size=args.batch_size,
                                             use_cpu=args.cpu, report_to="none"),
                      data_collator=CausalCollator(tokenizer.pad_token_id),
                      preprocess_logits_for_metrics=loss_sums, compute_metrics=token_metrics)
    print(json.dumps(trainer.evaluate(dataset, metric_key_prefix="test"), indent=2))


def main(argv=None):
    # The new workflow uses repository modules and versioned catalogs. Legacy
    # commands remain available for the user's existing data and checkpoints.
    import sys
    actual = list(sys.argv[1:] if argv is None else argv)
    if actual and actual[0] == 'stories':
        try:
            from story_training import main as story_main
        except ImportError as exc:
            raise RuntimeError('The stories workflow needs the repository checkout. In Colab, clone '
                               'christmasmodel and %cd into it before running model.py.') from exc
        return story_main(actual[1:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "train", "train-continuity", "train-coherence", "finetune-winter", "finetune-scenes", "compare-capacity", "compare-dropout", "compare-windowing", "compare-depth", "diagnose",
                                            "compare-decoding", "evaluate", "generate"])
    parser.add_argument("--corpus", help="Optional subset of the embedded Christmas URLs, as a Python sources list")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int,
                        help="Initialization/dropout seed; defaults to --seed. Change only this for a new seed replicate")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Require cached Gutenberg downloads")
    parser.add_argument("--min-success-rate", type=float, default=None)
    parser.add_argument("--output-dir")
    parser.add_argument("--model-dir", help="Saved project model; required for diagnose/compare-decoding")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attention-dropout", type=float, default=None,
                        help="Attention dropout for fresh models (capacity: 0.05; train: 0.1)")
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--batch-size", type=int, help="Microbatch: coherence 4, other commands 8")
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=float, default=None, help="Capacity comparison: 40; ordinary train: 12")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional optimizer-step cap; scene/winter pilots: 1000")
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--sample-tokens", type=int, default=96)
    parser.add_argument("--patience", type=int, default=6, help="Validation checks without a meaningful improvement")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", nargs="?", const="latest", help="Resume latest or a checkpoint directory")
    restart.add_argument("--continue-from", help="Start a new stage from this project's best/ or checkpoint weights; fresh optimizer/schedule")
    parser.add_argument("--grow-layers", type=int,
                        help="With --continue-from, expand to this depth by inserting initially identity GPT-2 blocks")
    parser.add_argument("--continuation-dropout", type=float,
                        help="Explicit new-stage override for embedding, residual, and attention dropout; otherwise inherit")
    parser.add_argument("--skip-test", action="store_true", default=True,
                        help="Compatibility flag: training uses validation only; run evaluate explicitly for test")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny CPU model, 3 updates, limited data; not quality training")
    parser.add_argument("--train-limit", type=int, help="Diagnostic window cap; omitted for full training")
    parser.add_argument("--random-window-offsets", action="store_true", default=None,
                        help="Shift interior training windows per book/epoch; keep evaluation fixed")
    parser.add_argument("--benchmark-steps", type=int, default=None,
                        help="Capacity comparison: save weights and run the full coherence panel at this interval")
    parser.add_argument("--benchmark-prompts", type=int, default=20,
                        help="Capacity comparison: frozen, book-balanced validation prefixes")
    parser.add_argument("--benchmark-tokens", type=int, default=192,
                        help="Capacity comparison: maximum generated tokens per prefix/seed")
    parser.add_argument("--plan-only", action="store_true",
                        help="Capacity/scene/winter experiment: freeze data/benchmark/settings without training")
    parser.add_argument("--scene-min-tokens", type=int, default=192)
    parser.add_argument("--scene-max-tokens", type=int, default=448)
    parser.add_argument("--scenes-per-book", type=int, default=16)
    parser.add_argument("--scene-review", help="Optional edited scene_candidates.csv; only include/natural_end may change")
    parser.add_argument("--winter-cache", help="Winter source download cache; defaults to winter-data beside --data-dir")
    parser.add_argument("--winter-lookback", type=int,
                        help="finetune-winter: preceding context tokens without duplicate loss (default 128; 0 for ablation)")
    parser.add_argument('--coherence-context', type=int, choices=[512, 1024],
                        help='train-coherence: default 1024; train-continuity: inherit source context')
    parser.add_argument('--continuity-weight', type=float, default=.5,
                        help='train-continuity: same-passage preference loss weight, default 0.5')
    parser.add_argument('--online-preference-weight', type=float, default=.5,
                        help='train-continuity: source-vs-model rollout preference weight; 0 disables')
    parser.add_argument('--curriculum-fraction', type=float,
                        help='train-coherence: fraction of updates to anneal simplicity weights; 0 disables')
    parser.add_argument('--context-loss-weight', type=float,
                        help='train-coherence: true-vs-unrelated-context loss weight; 0 disables')
    parser.add_argument('--context-margin', type=float, default=.2)
    parser.add_argument('--unlikelihood-weight', type=float, default=.1,
                        help='train-coherence: generated repetition penalty weight; 0 disables')
    parser.add_argument('--unlikelihood-every', type=int, default=8)
    parser.add_argument('--unlikelihood-tokens', type=int, default=64)
    parser.add_argument('--rerank-candidates', type=int,
                        help='train-coherence/generate: candidate count (coherence defaults to 8)')
    parser.add_argument("--with-control", action="store_true",
                        help="finetune-winter: also run a same-start, same-update original-corpus control")
    parser.add_argument("--eval-limit", type=int, help="Diagnostic window cap; omitted for full evaluation")
    parser.add_argument("--diagnostic-windows", type=int, default=1024,
                        help="Maximum windows per train/validation split for diagnose")
    parser.add_argument("--prompt", default=PROMPTS[0])
    parser.add_argument("--decode-mode", default="both",
                        choices=["both", "greedy", "sample", "blocked_sample", "contrastive", "reranked"],
                        help="Decoding method for generate; both preserves the original greedy/sample output")
    args = parser.parse_args(argv)
    args.winter_lookback_explicit = args.winter_lookback is not None
    continuity = args.command == 'train-continuity'
    coherence = args.command in {'train-coherence', 'train-continuity'}
    capacity = args.command == "compare-capacity"
    scenes = args.command == "finetune-scenes"
    winter = args.command in {'finetune-winter', 'train-coherence', 'train-continuity'}
    if args.curriculum_fraction is None:
        args.curriculum_fraction = 0. if continuity else .4
    if args.context_loss_weight is None:
        args.context_loss_weight = 0. if continuity else .1
    if args.coherence_context is None and not continuity:
        args.coherence_context = 1024
    if args.batch_size is None:
        args.batch_size = 4 if coherence else 8
    if coherence:
        args.block_size = args.coherence_context or 512
        if args.output_dir is None:
            args.output_dir = str(ROOT/('runs/christmas-continuity' if continuity else 'runs/christmas-coherence'))
        if args.learning_rate is None:
            args.learning_rate = 2e-5 if continuity else 3e-5
        if continuity and args.max_steps is None:
            args.max_steps = 1500
        if args.winter_lookback is None:
            args.winter_lookback = 256 if args.coherence_context == 1024 else 128
        args.rerank_candidates = 8 if args.rerank_candidates is None else args.rerank_candidates
    if (not 0 <= args.online_preference_weight <= 2 or not 0 <= args.continuity_weight <= 2
            or not 0 <= args.curriculum_fraction <= 1 or not 0 <= args.context_loss_weight <= 1
            or not 0 <= args.unlikelihood_weight <= 1 or not 0 <= args.context_margin <= 2
            or args.unlikelihood_every < 1 or not 8 <= args.unlikelihood_tokens <= 256
            or (args.rerank_candidates is not None and not 1 <= args.rerank_candidates <= 16)):
        parser.error('Invalid coherence setting: fractions/weights 0..1, margin 0..2, interval >=1, '
                     'generation length 8..256, candidates 1..16')
    defaults = {"min_success_rate": (1., .9), "dropout": (.05, .15),
                "attention_dropout": (.05, .1), "warmup_ratio": (.02, .05),
                "epochs": (40., 12.), "sample_steps": (500, 250),
                "random_window_offsets": (True, False),
                "output_dir": (str(ROOT / "runs" / "christmas-capacity"), str(ROOT / "runs" / "christmas"))}
    if scenes or winter:
        for name, value in {"output_dir": str(ROOT/("runs/christmas-winter-v2" if winter else "runs/christmas-scenes")), "learning_rate": 5e-5,
                            "warmup_ratio": .03, "max_steps": 1000, "benchmark_steps": 250,
                            "min_success_rate": 1., "random_window_offsets": True}.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
    for name, values in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, values[0 if capacity else 1])
    for name, value in {"learning_rate": 3e-4, "max_steps": -1, "benchmark_steps": 2000}.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.model_seed is None:
        args.model_seed = args.seed
    if not all(0 <= seed < 2**32 for seed in (args.seed, args.model_seed)):
        parser.error("Data and model seeds must be integers in [0, 2**32)")
    if args.plan_only and not (capacity or scenes or winter):
        parser.error("--plan-only is for compare-capacity, finetune-scenes or finetune-winter")
    if (args.winter_cache or args.with_control) and not winter:
        parser.error("--winter-cache and --with-control are for finetune-winter")
    if args.winter_lookback is not None and not winter:
        parser.error("--winter-lookback is for finetune-winter")
    if winter:
        args.winter_lookback = 128 if args.winter_lookback is None else args.winter_lookback
        if not 0 <= args.winter_lookback <= 256:
            parser.error("--winter-lookback must be between 0 and 256")
    if args.scene_review and not scenes:
        parser.error("--scene-review is for finetune-scenes")
    if scenes and (not 128 <= args.scene_min_tokens <= args.scene_max_tokens <= 510 or args.scenes_per_book <= 0):
        parser.error("Scenes need 128 <= minimum <= maximum <= 510 tokens, and a positive per-book cap")
    if capacity and (args.layers, args.width, args.heads) != (6, 384, 6):
        parser.error("compare-capacity fixes both architectures; do not override --layers/--width/--heads")
    if (capacity or scenes or winter) and (min(args.benchmark_steps, args.benchmark_prompts, args.benchmark_tokens) <= 0
                     or args.benchmark_tokens+160 > args.block_size):
        parser.error("Capacity benchmark counts must be positive and generation tokens + 160 must fit the context")
    if args.heads <= 0 or args.width % args.heads or args.block_size < 8 or args.vocab_size < 258:
        parser.error("width must divide by heads; block-size >= 8; vocab-size >= 258")
    if not 0 < args.min_success_rate <= 1 or not 0 <= args.dropout < 1 or not 0 <= args.attention_dropout < 1:
        parser.error("min-success-rate must be in (0,1]; both dropout rates in [0,1)")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("warmup-ratio must be in [0,1)")
    if args.continuation_dropout is not None and (
        not args.continue_from or not 0 <= args.continuation_dropout < 1
    ):
        parser.error("continuation-dropout requires --continue-from and a value in [0,1)")
    if args.grow_layers is not None and (
        not args.continue_from or args.grow_layers <= 0 or args.command not in {"train", "compare-depth"}
    ):
        parser.error("grow-layers requires train/compare-depth with --continue-from and a positive target depth")
    if any(x <= 0 for x in [args.layers, args.heads, args.width, args.batch_size, args.accumulation,
                            args.eval_steps, args.sample_steps, args.sample_tokens, args.patience,
                            args.epochs, args.learning_rate]):
        parser.error("Model sizes, batch sizes, intervals, epochs, and learning rate must be positive")
    if any(x is not None and x <= 0 for x in [args.train_limit, args.eval_limit]):
        parser.error("Diagnostic limits must be positive")
    if args.diagnostic_windows <= 0:
        parser.error("diagnostic-windows must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("max-steps must be -1 (use epochs) or a positive step budget")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        training(args)
    elif args.command == "finetune-scenes":
        finetune_scenes(args)
    elif args.command in {'finetune-winter', 'train-coherence', 'train-continuity'}:
        finetune_winter(args)
    elif args.command == "compare-capacity":
        compare_capacity(args)
    elif args.command == "compare-dropout":
        compare_dropout(args)
    elif args.command == "compare-windowing":
        compare_windowing(args)
    elif args.command == "compare-depth":
        compare_depth(args)
    elif args.command == "diagnose":
        diagnose(args)
    elif args.command == "compare-decoding":
        compare_decoding(args)
    else:
        inference(args)


def run_notebook():
    """Bootstrap a pasted cell without reading requirements.txt or kernel CLI flags."""
    import importlib
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *NOTEBOOK_REQUIREMENTS], check=True)
    importlib.invalidate_caches()
    main(NOTEBOOK_ARGS)


if __name__ == "__main__":
    if "__file__" in globals():
        main()  # python model.py ... or %run model.py ...
    else:
        run_notebook()  # Pasted as one complete notebook cell.
