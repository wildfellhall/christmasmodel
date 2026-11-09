"""Standalone Colab-ready GPT-2 trained from random weights on Gutenberg books.

Single-cell Colab usage: select a fresh GPU runtime, paste this entire file into
one code cell, and run it. Dependencies are installed and training starts using
the source list embedded below. No repository clone or corpus.py is required.
Edit NOTEBOOK_ARGS below to change options or resume a notebook run.

Colab GPU runtime, separate cells:
    !git clone https://github.com/wildfellhall/christmasmodel.git
    %cd /content/christmasmodel
    %pip install -q -r requirements.txt
    !python model.py train
    !python model.py generate --prompt "On Christmas Eve, the snow"

No pretrained model or tokenizer is used. --corpus can override the embedded list.
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
NOTEBOOK_ARGS = ["train"]  # For example: ["train", "--resume"] or ["train", "--smoke-test"].
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
    """One-token overlap preserves every next-token target, including book tails.

    Windows never cross books. Real EOS labels stay supervised; only padding is
    -100. No random-window split, and no synthetic next-book transitions.
    """
    def __init__(self, path, block_size, limit=None):
        import numpy as np
        self.books, self.windows = [], []
        self.block_size = block_size
        for line in Path(path).read_text().splitlines():
            book = json.loads(line)
            ids = np.asarray(book["ids"], dtype=np.int32)
            index = len(self.books)
            self.books.append(ids)
            self.windows.extend((index, start) for start in range(0, len(ids)-1, block_size-1))
        if limit:
            # Spread smoke/evaluation subsets across the corpus, not just book 1.
            indices = np.linspace(0, len(self.windows)-1, min(limit, len(self.windows)), dtype=int)
            self.windows = [self.windows[i] for i in indices]
        if not self.windows:
            raise ValueError(f"Empty token dataset: {path}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        book, start = self.windows[index]
        return {"input_ids": self.books[book][start:start+self.block_size].tolist()}


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
            labels[i, :n] = ids[i, :n]
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


def generate_samples(model, tokenizer, prompts, max_new_tokens=96, seed=1234):
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
                ids = tokenizer.encode(prompt, add_special_tokens=False) or [tokenizer.eos_token_id]
                room = model.config.n_positions - len(ids)
                if room <= 0:
                    raise ValueError("Prompt fills the context window; shorten it")
                inputs = torch.tensor([ids], dtype=torch.long, device=device)
                for mode in ("greedy", "sample"):
                    kwargs = dict(max_new_tokens=min(max_new_tokens, room),
                                  do_sample=mode == "sample", use_cache=True,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                  suppress_tokens=[tokenizer.pad_token_id])
                    if mode == "sample":
                        kwargs.update(temperature=.8, top_p=.92, top_k=40, repetition_penalty=1.08)
                    output = model.generate(inputs, attention_mask=torch.ones_like(inputs), **kwargs)
                    results.append({"prompt": prompt, "mode": mode,
                                    "continuation": tokenizer.decode(output[0, len(ids):], skip_special_tokens=True)})
    finally:
        model.train(was_training)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
    return results


def dataset_signature(data):
    digest = hashlib.sha256()
    for name in ("report.json", "train.jsonl", "validation.jsonl", "test.jsonl", "tokenizer/tokenizer.json"):
        with (Path(data) / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def training(args):
    import torch
    from transformers import (
        AutoTokenizer,
        EarlyStoppingCallback,
        GPT2Config,
        GPT2LMHeadModel,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint
    if args.resume and args.rebuild:
        raise ValueError("Do not rebuild a dataset while resuming an existing run")
    report = prepare(args)
    if args.smoke_test:
        args.layers, args.width, args.heads, args.block_size = 2, 64, 2, 64
        args.batch_size, args.accumulation, args.max_steps = 2, 1, 3
        args.eval_steps, args.sample_steps, args.sample_tokens = 1, 1, 12
        args.train_limit, args.eval_limit, args.cpu = 16, 8, True
    set_seed(args.seed)
    data, out = Path(args.data_dir), Path(args.output_dir)
    resume = None
    if args.resume:
        resume = get_last_checkpoint(str(out)) if args.resume == "latest" else str(Path(args.resume).resolve())
        if not resume:
            raise ValueError("No checkpoint found to resume")
        if Path(resume).resolve().parent != out.resolve():
            raise ValueError("Resume checkpoint must belong to this --output-dir")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; use --resume or a new --output-dir")
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    tokenizer.model_max_length = args.block_size
    signature = dataset_signature(data)
    settings = {"dataset_sha256": signature, "layers": args.layers, "width": args.width,
                "heads": args.heads, "block_size": args.block_size, "dropout": args.dropout,
                "batch_size": args.batch_size, "accumulation": args.accumulation,
                "learning_rate": args.learning_rate, "epochs": args.epochs, "max_steps": args.max_steps,
                "seed": args.seed, "train_limit": args.train_limit, "eval_limit": args.eval_limit,
                "eval_steps": args.eval_steps, "patience": args.patience, "smoke_test": args.smoke_test}
    if resume:
        prior = json.loads((out / "run_config.json").read_text())
        if prior != settings:
            raise ValueError("Run settings or dataset differ from saved run_config.json; use a new run")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", settings)
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=args.block_size, n_ctx=args.block_size,
                        n_embd=args.width, n_layer=args.layers, n_head=args.heads, n_inner=4*args.width,
                        activation_function="gelu_new", resid_pdrop=args.dropout,
                        embd_pdrop=args.dropout, attn_pdrop=.1, initializer_range=.02,
                        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id, tie_word_embeddings=True, use_cache=False)
    # This constructor initializes ALL weights randomly. Never load pretrained GPT-2.
    model = GPT2LMHeadModel(config)
    parameters = model.num_parameters()
    train_set = BookWindows(data / "train.jsonl", args.block_size, args.train_limit)
    val_set = BookWindows(data / "validation.jsonl", args.block_size, args.eval_limit)
    effective_batch = args.batch_size * args.accumulation
    print(f"Random GPT-2: {parameters:,} parameters; {len(train_set):,} training windows; "
          f"effective batch {effective_batch} (~{effective_batch*(args.block_size-1):,} targets/update).", flush=True)
    print(f"Unique training target tokens/parameter: {report['splits']['train']['target_tokens']/parameters:.2f}. "
          "Repeated epochs do not add new language evidence.", flush=True)
    use_cuda = torch.cuda.is_available() and not args.cpu
    bf16 = use_cuda and torch.cuda.is_bf16_supported()

    class SampleCallback(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            if state.global_step % args.sample_steps == 0 and state.is_world_process_zero:
                samples = generate_samples(kwargs["model"], tokenizer, PROMPTS, args.sample_tokens)
                with (out / "samples.jsonl").open("a", encoding="utf-8") as handle:
                    for sample in samples:
                        handle.write(json.dumps({"step": state.global_step, **sample}, ensure_ascii=False) + "\n")
                        print(f"\n[step {state.global_step} | {sample['mode']}] "
                              f"{sample['prompt']}{sample['continuation']}\n", flush=True)

    training_args = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation, learning_rate=args.learning_rate,
        weight_decay=.1, adam_beta1=.9, adam_beta2=.95, adam_epsilon=1e-8,
        max_grad_norm=1., warmup_ratio=.05, lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": .1}, optim="adamw_torch",
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        logging_steps=min(25, args.eval_steps), save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="nll", greater_is_better=False,
        fp16=use_cuda and not bf16, bf16=bf16, use_cpu=args.cpu,
        dataloader_num_workers=0, dataloader_pin_memory=use_cuda,
        seed=args.seed, data_seed=args.seed, report_to="none", save_safetensors=True,
        restore_callback_states_from_checkpoint=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=val_set,
                      data_collator=CausalCollator(tokenizer.pad_token_id), processing_class=tokenizer,
                      preprocess_logits_for_metrics=loss_sums, compute_metrics=token_metrics,
                      callbacks=[SampleCallback(), EarlyStoppingCallback(early_stopping_patience=args.patience,
                                                                         early_stopping_threshold=.002)])
    planned_steps = args.max_steps if args.max_steps > 0 else math.ceil(len(train_set)/effective_batch)*args.epochs
    if planned_steps < args.eval_steps:
        raise ValueError("Training would finish before its first evaluation; lower --eval-steps")
    if resume and json.loads((Path(resume) / "trainer_state.json").read_text())["global_step"] >= planned_steps:
        raise ValueError("This run already reached its step budget; use generate/evaluate or start a new run")
    trainer.train(resume_from_checkpoint=resume)
    # Trainer has restored the best VALIDATION checkpoint, not necessarily the last.
    model.config.use_cache = True
    best = out / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(best)
    trainer.save_state()
    trainer.remove_callback(EarlyStoppingCallback)
    test_set = BookWindows(data / "test.jsonl", args.block_size, args.eval_limit)
    metrics = trainer.evaluate(test_set, metric_key_prefix="test")
    summary = {"parameters": parameters, "best_checkpoint": trainer.state.best_model_checkpoint,
               "best_validation_nll": trainer.state.best_metric, "global_step": trainer.state.global_step,
               "dataset_sha256": signature, "smoke_test": args.smoke_test,
               "limited_data": bool(args.train_limit or args.eval_limit), "test": metrics,
               "corpus": report, "run": settings}
    write_json(best / "training_summary.json", summary)
    write_json(out / "test_metrics.json", metrics)
    final_samples = generate_samples(model, tokenizer, PROMPTS, args.sample_tokens)
    write_json(best / "samples.json", final_samples)
    (best / "README.md").write_text(model_card(summary), encoding="utf-8")
    print(f"Best model and tokenizer: {best}\nHeld-out test: {json.dumps(metrics, indent=2)}", flush=True)


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

A {summary['parameters']:,}-parameter GPT-2 language model initialized randomly
and trained on the Gutenberg works listed in
[christmasmodel](https://github.com/wildfellhall/christmasmodel/blob/main/corpus.py).
Its byte-level BPE vocabulary was learned on training books only.

This is a literary continuation model, not an instruction/chat model. Use a
story opening as the prompt. Coherence, novelty, and factual accuracy are not
guaranteed. The limited historical corpus can produce memorized passages,
repetition, dated language, and historical biases. Repeated epochs are not new data.

Best validation NLL: {summary['best_validation_nll']}
Test token perplexity: {summary['test'].get('test_perplexity')}
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
        for sample in generate_samples(model, tokenizer, [args.prompt], args.sample_tokens, args.seed):
            print(f"\n[{sample['mode']}] {sample['prompt']}{sample['continuation']}")
        return
    summary = json.loads((path / "training_summary.json").read_text())
    if summary["dataset_sha256"] != dataset_signature(args.data_dir):
        raise ValueError("Evaluation data/tokenizer differ from this model's training run")
    dataset = BookWindows(Path(args.data_dir) / "test.jsonl", model.config.n_positions, args.eval_limit)
    trainer = Trainer(model=model,
                      args=TrainingArguments(output_dir=str(path.parent / "evaluation"),
                                             per_device_eval_batch_size=args.batch_size,
                                             use_cpu=args.cpu, report_to="none"),
                      data_collator=CausalCollator(tokenizer.pad_token_id),
                      preprocess_logits_for_metrics=loss_sums, compute_metrics=token_metrics)
    print(json.dumps(trainer.evaluate(dataset, metric_key_prefix="test"), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "generate"])
    parser.add_argument("--corpus", help="Optional Python file containing sources; defaults to the embedded URLs")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Require cached Gutenberg downloads")
    parser.add_argument("--min-success-rate", type=float, default=.9)
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "christmas"))
    parser.add_argument("--model-dir", help="Saved model for evaluate/generate; defaults to output-dir/best")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=float, default=12)
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional optimizer-step cap")
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--sample-steps", type=int, default=250)
    parser.add_argument("--sample-tokens", type=int, default=96)
    parser.add_argument("--patience", type=int, default=6, help="Validation checks without a meaningful improvement")
    parser.add_argument("--resume", nargs="?", const="latest", help="Resume latest or a checkpoint directory")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny CPU model, 3 updates, limited data; not quality training")
    parser.add_argument("--train-limit", type=int, help="Diagnostic window cap; omitted for full training")
    parser.add_argument("--eval-limit", type=int, help="Diagnostic window cap; omitted for full evaluation")
    parser.add_argument("--prompt", default=PROMPTS[0])
    args = parser.parse_args(argv)
    if args.heads <= 0 or args.width % args.heads or args.block_size < 8 or args.vocab_size < 258:
        parser.error("width must divide by heads; block-size >= 8; vocab-size >= 258")
    if not 0 < args.min_success_rate <= 1 or not 0 <= args.dropout < 1:
        parser.error("min-success-rate must be in (0,1]; dropout in [0,1)")
    if any(x <= 0 for x in [args.layers, args.heads, args.width, args.batch_size, args.accumulation,
                            args.eval_steps, args.sample_steps, args.sample_tokens, args.patience,
                            args.epochs, args.learning_rate]):
        parser.error("Model sizes, batch sizes, intervals, epochs, and learning rate must be positive")
    if any(x is not None and x <= 0 for x in [args.train_limit, args.eval_limit]):
        parser.error("Diagnostic limits must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("max-steps must be -1 (use epochs) or a positive step budget")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        training(args)
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
