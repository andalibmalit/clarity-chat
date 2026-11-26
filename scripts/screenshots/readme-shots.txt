# The README screenshots chat.png, details.png and compare.png (SmolLM2, made-up text);
# settings.png is in readme-settings.txt. Details' record section stays closed.
sleep 3
click role=menuitemradio[name=/^smol-chat Tiny model.*llama.cpp/]
wait role=textbox[name="Message"]
sleep 5
fill role=textbox[name="Message"] :: Name three colours you might see in a sunset.
click role=button[name="Send"]
wait text=Regenerate
sleep 2
shot chat
click role=button[name="Details"]
sleep 2
shot details
click #panel:not([hidden]) [data-action=panel-close]
sleep 1
click role=button[name="Regenerate"]
wait role=button[name=/^Replies 2/]
sleep 2
click role=button[name=/^Settings/]
sleep 2
# settings.png comes from readme-settings.txt instead: pressing End here would bring the
# Context row's memory figure (a machine fact) into view.
fill #set-temperature :: 1.2
press Enter
sleep 2
click #panel:not([hidden]) [data-action=panel-close]
sleep 1
click role=button[name=/^Replies 2/]
sleep 1
click .listpop [data-action=again]
wait role=button[name=/^Replies 4/]
sleep 2
click role=button[name=/^Replies 4/]
sleep 1
click .listpop [data-action=compare]
wait #compare:not([hidden]) .cmp-grid
sleep 2
shot compare
