import java.util.*;
import java.io.*;

public class Generator2 {
    private int k = 10;
    private int layers = 1;
    private int n_prec = 7;
    private int v_prec = 5;
    private int t_prec = 16;

    private char n_f, v_f, t0_f, t1_f;

    private String data_path = "data";

    private static final long ZERO_TIME = (1 << 16) - (1 << 8);

    private String seed;
    private Map<String, List<String>> data;

    public Generator2(String[] args) { 
        read_args(args);
        data = new HashMap<>();
    }

    private void read_args(String[] args) {
        for (String arg: args) {
            if (arg.startsWith("k=")) {
                k = Integer.parseInt(arg.substring(2));
            }
            else if (arg.startsWith("l=")) {
                layers = Integer.parseInt(arg.substring(2));
            }
            else if (arg.startsWith("prec=")) {
                arg = arg.substring(5);
                String[] dat = arg.split(",");
                n_prec = Integer.parseInt(dat[0]);
                v_prec = Integer.parseInt(dat[1]);
                t_prec = Integer.parseInt(dat[2]);
            }
            else if (arg.startsWith("path=")) {
                data_path = arg.substring(5);
            }
        }

        n_f = (char) (256 - (1 << (7 - n_prec)));
        v_f = (char) (256 - (1 << (7 - v_prec)));
        int t_f = (short) (65536 - (1 << (16 - t_prec)));
        t0_f = (char) ((t_f >> 8) & 255);
        t1_f = (char) (t_f & 255);
    }

    public void read_data() throws IOException {
        Scanner sc = new Scanner(new BufferedReader(new InputStreamReader(new FileInputStream(data_path), "UTF-8")));
        int num_tracks = Math.min(sc.nextInt(), 10);
        data.put("", new ArrayList<>());
        for (int t=0; t < num_tracks; t++) {
            int len = sc.nextInt();
            String cur = "";
            for (int i = 0; i < k; i++) {
                cur += encode(sc.nextInt());
            }
            for (int i = k; i < len; i++) {
                
                if (!data.containsKey(cur)) data.put(cur, new ArrayList<>());
                String note = encode(sc.nextInt());
                data.get(cur).add(note);
                data.get("").add(note);
                cur = cur.substring(4) + note;
            }
        }
        sc.close();
    }

    private String encode(int note) {
        return new String(new char[] {
            // (char) (br.read() & n_f),
            // (char) (br.read() & v_f),
            // (char) (br.read() & t0_f),
            // (char) (br.read() & t1_f),

            (char) ((note >> 24) & n_f),
            (char) ((note >> 16) & v_f),
            (char) ((note >> 8) & t0_f),
            (char) ((note >> 0) & t1_f),

            // (char) (note >> 24),
            // (char) (note >> 16),
            // (char) (note >> 8),
            // (char) (note >> 0),
        });
    }

    private static int decode(String note) {
        return ((int) (note.charAt(0)) << 24) | ((int) (note.charAt(1)) << 16) | ((int) (note.charAt(2)) << 8) | ((int) (note.charAt(3)) << 0);
    }

    public void run() throws IOException {
        PrintWriter pw = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out, "UTF-8")));
        
        try {
            while (true) {
                List<String> choices = data.get(seed);
                if (choices == null || choices.size() < 2) choices = data.get("");
                for (int i = 0; i < layers; i++) {
                    String note = choices.get((int) (choices.size() * Math.random()));
                    if (i == 0) pw.println("" + decode(note));
                    else pw.println(decode(note) & ZERO_TIME);
                }
            }
        }
        finally {
            pw.close();
        }
    }

    // private static List<String> get_all_files(String path, String ext) {
    //     if (path == null) path = ".";
    //     if (ext == null) ext="";
    //     List<String> result = new ArrayList<>();
    //     return get_all_files(path, ext, result);
    // }
    // private static List<String> get_all_files(String path, String ext, List<String> result) {
    //     File cur = new File(path);
    //     if (cur.isDirectory()) {
    //         for (String next: cur.list()) {
    //             get_all_files(path + next, ext, result);
    //         }
    //     }
    //     else if (cur.isFile() && cur.getName().endsWith(ext)) {
    //         result.add(path);
    //     }
    //     return result;
    // }

    public static void main(String[] args) throws IOException {
        Generator2 g = new Generator2(args);
        g.read_data();
        g.run();
        
        // for (int i = 0; i < 1000000; i++) {
        //     int n = (int) (Integer.MAX_VALUE * Math.random());
        //     if (decode(encode(n)) != n) System.out.println("ERROR");
        // }
    }
}