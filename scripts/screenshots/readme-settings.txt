# The README's settings.png (SmolLM2, made-up text). Settings opens with a click and is
# scrolled only to the Documents group, by a click on its intro line, so the Context row's
# memory figure (a machine fact) stays out of view. The click is scoped to the open panel,
# because a hidden popover holds the same line.
sleep 3
click role=menuitemradio[name=/^smol-chat Tiny model.*llama.cpp/]
wait role=textbox[name="Message"]
sleep 5
fill role=textbox[name="Message"] :: Name three colours you might see in a sunset.
click role=button[name="Send"]
wait text=Regenerate
sleep 2
click role=button[name=/^Settings/]
sleep 2
click #panel:not([hidden]) p:has-text("A change applies from your next message")
sleep 1
shot settings
