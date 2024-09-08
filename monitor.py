import mido

port_name = '16n Port 1'

# Try to open the specified MIDI input port
try:
    with mido.open_input(port_name) as port:
        print(f"Listening for Control Change (CC) messages on port: {port_name}")

        # Loop to monitor incoming MIDI messages
        for message in port:
            # Check if the message is a Control Change (CC) message
            if message.type == 'control_change':
                # Print the CC message in the desired format
                print(f"cc {message.control} {message.value}")

except IOError:
    print(f"Error: Could not open MIDI port '{port_name}'. Make sure the port name is correct and the device is connected.")