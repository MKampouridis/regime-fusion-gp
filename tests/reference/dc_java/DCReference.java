import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Reference output of the group's earlier Java DC implementation.
 *
 * generateEvents, detect2 and adjust are copied from misc/DCEventGenerator.java in
 * github.com/adesolaadegboye/SymbolicRegression (commit af564a5). Only three things
 * changed: the Event class is reduced to the fields this logic uses, the enum
 * constants are written Type.Upturn etc. (a single file cannot use static
 * imports), and the closing summary statistics and progress print are dropped.
 *
 * Usage: java DCReference <file with one value per line> <delta>
 * Prints one line per DC event: type,start,end (1-based, as in the original).
 */
public class DCReference {

    enum Type { Upturn, Downturn, UpwardOvershoot, DownwardOvershoot }

    static class Event {
        int start = 0;
        int end = 0;
        Type type;
        Event overshoot;
        double DCValue = 0.0;

        Event(int start, int end, Type type) {
            this.start = start;
            this.end = end;
            this.type = type;
        }

        Event(int start, int end, Type type, double value) {
            this.start = start;
            this.end = end;
            this.type = type;
            this.DCValue = value;
        }
    }

    Event[] output;
    Event[] generatedEvents;

    public void generateEvents(Double[] values, double delta) {
        ArrayList<Event> events = new ArrayList<Event>();
        Type event = Type.Upturn;

        Event last = new Event(0, 0, Type.Upturn);
        events.add(last);

        output = new Event[values.length];

        double pHigh = 0;
        double pLow = 0;

        int[] indexDC = new int[2]; // DC event indexes
        int[] indexOS = new int[2]; // OS event indexes
        int index = 1;

        for (double value : values) {
            if (index == 1) {
                // it is the first line

                pHigh = value;
                pLow = value;

                Arrays.fill(indexDC, index);
                Arrays.fill(indexOS, index);
            } else if (event == Type.Upturn) {
                if (value <= (pHigh * (1 - delta))) {
                    last.overshoot = detect2(Type.UpwardOvershoot, indexDC, indexOS, value);

                    adjust(last.overshoot == null ? last : last.overshoot, indexOS, indexOS);

                    event = Type.Downturn;
                    double percentageDisplacement = Math.abs(pHigh - pLow) / 100.0;
                    pLow = value;

                    indexDC[1] = index;
                    indexOS[0] = index + 1;

                    last = new Event(indexDC[0], indexDC[1], Type.Downturn, percentageDisplacement);
                    events.add(last);
                } else if (pHigh < value) {
                    pHigh = value;

                    indexDC[0] = index;
                    indexOS[1] = index - 1;
                }
            } else {
                if (value >= (pLow * (1 + delta))) {
                    last.overshoot = detect2(Type.DownwardOvershoot, indexDC, indexOS, value);

                    adjust(last.overshoot == null ? last : last.overshoot, indexDC, indexOS);
                    double percentageDisplacement = Math.abs(pHigh - pLow) / 100.0;
                    event = Type.Upturn;
                    pHigh = value;

                    indexDC[1] = index;
                    indexOS[0] = index + 1;

                    last = new Event(indexDC[0], indexDC[1], Type.Upturn, percentageDisplacement);
                    events.add(last);
                } else if (pLow > value) {
                    pLow = value;

                    indexDC[0] = index;
                    indexOS[1] = index - 1;
                }
            }

            output[index - 1] = last;

            index++;
        }
        this.generatedEvents = events.toArray(new Event[events.size()]);
    }

    protected Event detect2(Type type, int[] indexDC, int[] indexOS, double value) {
        // overshoot event must have a start index lower that
        // the DC event start index
        if (indexOS[0] < indexOS[1] && indexOS[0] < indexDC[0]) {
            return new Event(indexOS[0], indexOS[1], type, value);
        }

        return null;
    }

    protected void adjust(Event last, int[] indexDC, int[] indexOS) {
        // we might miss the start of an event
        if (indexDC[0] == last.start) {
            indexDC[0] = last.end + 1;
        }
        // we might skip the start of an event when there
        // are repeated values or large increases during an
        // upturn overshoot followed by a downturn event and
        // vice-versa (the overshoot will be invalid since
        // the end index will be smaller than the start index)
        else if (indexDC[0] > (last.end + 1)) {
            indexDC[0] = (last.end + 1);
        }
    }

    public static void main(String[] args) throws IOException {
        ArrayList<Double> list = new ArrayList<Double>();
        try (BufferedReader reader = new BufferedReader(new FileReader(args[0]))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    list.add(Double.parseDouble(line));
                }
            }
        }
        double delta = Double.parseDouble(args[1]);
        DCReference generator = new DCReference();
        generator.generateEvents(list.toArray(new Double[0]), delta);
        StringBuilder out = new StringBuilder();
        // The first entry is the placeholder Event(0, 0, Upturn) created before the loop.
        for (int i = 1; i < generator.generatedEvents.length; i++) {
            Event e = generator.generatedEvents[i];
            out.append(e.type).append(',').append(e.start).append(',').append(e.end).append('\n');
        }
        System.out.print(out);
    }
}
