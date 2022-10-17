function debugLog(message) {
    console.log("[FetchFS] " + message);
}

const MODE_DIR = 16895;
const MODE_FILE = 33206;

FETCHFS = {
    mount: function(mount) {
        debugLog("Mounting FetchFS");
        console.log(mount);
        return this.createNode(null, "/", MODE_DIR);
    },
    createNode: function(parent, name, mode) {
        debugLog("Creating Node");

        let node = FS.createNode(parent, name, mode);
        node.node_ops = {
            lookup: (parent, name) => {
                console.log("Lookup " + name + " in " + parent);
                return this.createNode(parent, name, MODE_FILE);
            }
        };
        node.stream_ops = {

        };
        return node;
    }
}
