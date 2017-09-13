    //      openmp cleanup to be triggered at bohrium runtime.

    // for(void *handle: _lib_handles) {
    //     dlerror(); // Reset errors
    //     if (dlclose(handle)) {
    //         cerr << dlerror() << endl;
    //     }
    // }
}

KernelFunction EngineFortran::getFunction(const string &source) {
    size_t hash = hasher(source);
    ++stat.kernel_cache_lookups;

    // Do we have the function compiled and ready already?
    if (_functions.find(hash) != _functions.end()) {
        return _functions.at(hash);
    }

    fs::path binfile = cache_bin_dir / jitk::hash_filename(compilation_hash, hash, ".so");

    // If the binary file of the kernel doesn't exist we create it
    if (verbose or cache_bin_dir.empty() or not fs::exists(binfile)) {
        ++stat.kernel_cache_misses;

        // We create the binary file in the tmp dir
        binfile = tmp_bin_dir / jitk::hash_filename(compilation_hash, hash, ".so");

        // Write the source file and compile it (reading from disk)
        // NB: this is a nice debug option, but will hurt performance
        if (verbose) {
            fs::path srcfile = jitk::write_source2file(source, tmp_src_dir,
                                                       jitk::hash_filename(compilation_hash, hash, ".f95"),
                                                       true);
            compiler.compile(binfile.string(), srcfile.string());
        } else {
            // Pipe the source directly into the compiler thus no source file is written
            compiler.compile(binfile.string(), source.c_str(), source.size());
        }
    }

    // Load the shared library
    void *lib_handle = dlopen(binfile.string().c_str(), RTLD_NOW);
    if (lib_handle == nullptr) {
        cerr << "Cannot load library: " << dlerror() << endl;
        throw runtime_error("VE-FORTRAN: Cannot load library");
    }
    _lib_handles.push_back(lib_handle);

    // Load the launcher function
    // The (clumsy) cast conforms with the ISO C standard and will
    // avoid any compiler warnings.
    dlerror(); // Reset errors
    *(void **) (&_functions[hash]) = dlsym(lib_handle, "launcher_");
    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr) {
        cerr << "Cannot load function launcher(): " << dlsym_error << endl;
        throw runtime_error("VE-FORTRAN: Cannot load function launcher()");
    }
    return _functions.at(hash);
}


void EngineFortran::execute(const std::string &source, const std::vector<bh_base*> &non_temps,
                           const std::vector<const bh_view*> &offset_strides,
                           const std::vector<const bh_instruction*> &constants) {

    // Make sure all arrays are allocated
    for (bh_base *base: non_temps) {
        bh_data_malloc(base);
    }

    // Compile the kernel
    auto tbuild = chrono::steady_clock::now();
    KernelFunction func = getFunction(source);
    assert(func != NULL);
    stat.time_compile += chrono::steady_clock::now() - tbuild;

    // Create a 'data_list' of data pointers
    vector<void*> data_list;
    data_list.reserve(non_temps.size());
    for(bh_base *base: non_temps) {
        assert(base->data != NULL);
        data_list.push_back(base->data);
    }

    // And the offset-and-strides
    vector<uint64_t> offset_and_strides;
    offset_and_strides.reserve(offset_strides.size());
    for (const bh_view *view: offset_strides) {
        const uint64_t t = (uint64_t) view->start;
        offset_and_strides.push_back(t);
        for (int i=0; i<view->ndim; ++i) {
            const uint64_t s = (uint64_t) view->stride[i];
            offset_and_strides.push_back(s);
        }
    }

    // And the constants
    vector<bh_constant_value> constant_arg;
    constant_arg.reserve(constants.size());
    for (const bh_instruction* instr: constants) {
        constant_arg.push_back(instr->constant.value);
    }

    auto texec = chrono::steady_clock::now();
    // Call the launcher function, which will execute the kernel
    func(&data_list[0], &offset_and_strides[0], &constant_arg[0]);
    stat.time_exec += chrono::steady_clock::now() - texec;

}

void EngineFortran::set_constructor_flag(std::vector<bh_instruction*> &instr_list) {
    const std::set<bh_base*> empty;
    jitk::util_set_constructor_flag(instr_list, empty);
}


std::string EngineFortran::info() const {
    stringstream ss;
    ss << "----"                                                           << "\n";
    ss << "Fortran:"                                                        << "\n";
    ss << "  Hardware threads: " << std::thread::hardware_concurrency()    << "\n";
    ss << "  JIT Command: \"" << compiler.process_str("${OBJ}", "${SRC}")  << "\"\n";
    return ss.str();
}

} // bohrium
