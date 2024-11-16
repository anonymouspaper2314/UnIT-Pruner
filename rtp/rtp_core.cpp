#include <torch/all.h>
#include <torch/python.h>
#include <tuple>

/*
***********************************
LINEAR LAYERS
***********************************
*/

//https://github.com/pytorch/pytorch/blob/c9653bf2ca6dd88b991d71abf836bd9a7a1d9dc3/aten/src/ATen/native/LinearAlgebra.cpp#L1638

torch::Tensor linear(torch::Tensor &it, torch::Tensor &wt){
    unsigned int bs = it.size(0);
    unsigned int ps = it.size(1);
    unsigned int ns = wt.size(0);

    torch::Tensor ot = torch::empty({
        bs,
        ns
    });

    torch::TensorAccessor<float, 2> ia = it.accessor<float, 2>();
    torch::TensorAccessor<float, 2> wa = wt.accessor<float, 2>();
    torch::TensorAccessor<float, 2> oa = ot.accessor<float, 2>();

    float tmp;
    unsigned int bi, pi, ni;
    for(bi = 0; bi < bs; bi++){
        torch::TensorAccessor<float, 1> sa = ia[bi];
        for(ni = 0; ni < ns; ni++){
            tmp = 0;
            torch::TensorAccessor<float, 1> wra = wa[ni];

            for(pi = 0; pi < ps; pi++){
                tmp += sa[pi] * wra[pi];
            }

            oa[bi][ni] = tmp;
        }
    }

    return ot;
}

torch::Tensor rtp_linear(torch::Tensor &it, torch::Tensor &wt, float threshold){
    unsigned int bs = it.size(0);
    unsigned int ps = it.size(1);
    unsigned int ns = wt.size(0);

    torch::Tensor ot = torch::zeros({
        bs,
        ns
    });

    torch::TensorAccessor<float, 2> ia = it.accessor<float, 2>();
    torch::TensorAccessor<float, 2> wa = wt.accessor<float, 2>();
    torch::TensorAccessor<float, 2> oa = ot.accessor<float, 2>();

    float av, wv, t;
    unsigned int bi, pi, ni;
    for(bi = 0; bi < bs; bi++){
        torch::TensorAccessor<float, 1> sa = ia[bi];
        for(pi = 0; pi < ps; pi++){
            av = sa[pi];
            if(av != 0){
                t = fabs(threshold / av);
                for(ni = 0; ni < ns; ni++){
                    wv = wa[ni][pi];
                    if(wv > t || wv < -t){
                        oa[bi][ni] += av * wv;
                    }
                }
            }
        }
    }

    return ot;
}

std::tuple<at::Tensor, unsigned int, unsigned int, unsigned int> debug_rtp_linear(torch::Tensor &it, torch::Tensor &wt, float threshold){
    unsigned int bs = it.size(0);
    unsigned int ps = it.size(1);
    unsigned int ns = wt.size(0);

    torch::Tensor ot = torch::zeros({
        bs,
        ns
    });

    torch::TensorAccessor<float, 2> ia = it.accessor<float, 2>();
    torch::TensorAccessor<float, 2> wa = wt.accessor<float, 2>();
    torch::TensorAccessor<float, 2> oa = ot.accessor<float, 2>();

    unsigned int pruned_in_training = 0;
    unsigned int pruned_in_inference = 0;
    unsigned int total = bs * ps * ns;

    float av, t, wv;
    unsigned int bi, pi, ni;
    for(bi = 0; bi < bs; bi++){
        torch::TensorAccessor<float, 1> sa = ia[bi];
        for(pi = 0; pi < ps; pi++){
            av = sa[pi];
            if(av != 0){
                t = fabs(threshold / av);
                for(ni = 0; ni < ns; ni++){
                    wv = wa[ni][pi];
                    if(wv > t || wv < -t){
                        oa[bi][ni] += av * wv;
                    }else{
                        if(wv == 0){
                            pruned_in_training++;
                        }else{
                            pruned_in_inference++;
                        }
                    }
                }
            }else{
                for(ni = 0; ni < ns; ni++){
                    wv = wa[ni][pi];
                    if(wv == 0){
                        pruned_in_training++;
                    }else{
                        pruned_in_inference++;
                    }
                }
            }
        }
    }

    return std::make_tuple(ot, pruned_in_training, pruned_in_inference, total);
}

/*
***********************************
LINEAR LAYERS (END)
***********************************
*/



/*
***********************************
CONVOLUTIONAL 2D LAYERS
***********************************
*/

torch::Tensor conv2d(torch::Tensor &it, torch::Tensor &wt, unsigned int stride, unsigned int padding, unsigned int dilation){    
    it = torch::nn::functional::pad(
        it,
        torch::nn::functional::PadFuncOptions({padding, padding, padding, padding})
            .mode(torch::kConstant)
            .value(0)
    );
    
    unsigned int bs = it.size(0);
    unsigned int ic = it.size(1);
    unsigned int ys = it.size(2);
    unsigned int xs = it.size(3);
    unsigned int oc = wt.size(0);
    unsigned int ks = wt.size(2); //TODO: allow non-square convolutions

    unsigned int oys = ((ys - (((ks - 1) * dilation) + 1)) / stride) + 1;
    unsigned int oxs = ((xs - (((ks - 1) * dilation) + 1)) / stride) + 1;

    torch::Tensor ot = torch::zeros({
        bs,
        oc,
        oys,
        oxs
    });

    torch::TensorAccessor<float, 4> ia = it.accessor<float, 4>();
    torch::TensorAccessor<float, 4> wa = wt.accessor<float, 4>();
    torch::TensorAccessor<float, 4> oa = ot.accessor<float, 4>();

    float tmp;
    unsigned int bi, ici, oci, yi, xi, fyi, fxi;
    for(bi = 0; bi < bs; bi++){
        torch::TensorAccessor<float, 3> ocaa = oa[bi];
        torch::TensorAccessor<float, 3> sa = ia[bi];
        for(ici = 0; ici < ic; ici++){
            torch::TensorAccessor<float, 2> ca = sa[ici];
            for(oci = 0; oci < oc; oci++){
                torch::TensorAccessor<float, 2> oca = ocaa[oci];
                torch::TensorAccessor<float, 2> fa = wa[oci][ici];
                for(yi = 0; yi < oys; yi++){
                    torch::TensorAccessor<float, 1> ocra = oca[yi];
                    for(xi = 0; xi < oxs; xi++){
                        tmp = 0;

                        for(fyi = 0; fyi < ks; fyi++){
                            torch::TensorAccessor<float, 1> cra = ca[(yi * stride) + (fyi * dilation)];
                            torch::TensorAccessor<float, 1> fra = fa[fyi];
                            for(fxi = 0; fxi < ks; fxi++){
                                tmp += cra[(xi * stride) + (fxi * dilation)] * fra[fxi];
                            }
                        }

                        ocra[xi] += tmp;
                    }
                }
            }
        }
    }

    return ot;
}

torch::Tensor rtp_conv2d(torch::Tensor &it, torch::Tensor &wt, unsigned int stride, unsigned int padding, unsigned int dilation, float threshold){
    it = torch::nn::functional::pad(
        it,
        torch::nn::functional::PadFuncOptions({padding, padding, padding, padding})
            .mode(torch::kConstant)
            .value(0)
    );

    unsigned int bs = it.size(0);
    unsigned int ic = it.size(1);
    unsigned int ys = it.size(2);
    unsigned int xs = it.size(3);
    unsigned int oc = wt.size(0);
    unsigned int ks = wt.size(2); //TODO: allow non-square convolutions

    unsigned int oys = ((ys - (((ks - 1) * dilation) + 1)) / stride) + 1;
    unsigned int oxs = ((xs - (((ks - 1) * dilation) + 1)) / stride) + 1;

    torch::Tensor tt = torch::empty({
        ic,
        oc,
        ks,
        ks
    });

    torch::Tensor ot = torch::zeros({
        bs,
        oc,
        oys,
        oxs
    });

    torch::TensorAccessor<float, 4> ta = tt.accessor<float, 4>();
    torch::TensorAccessor<float, 4> ia = it.accessor<float, 4>();
    torch::TensorAccessor<float, 4> wa = wt.accessor<float, 4>();
    torch::TensorAccessor<float, 4> oa = ot.accessor<float, 4>();

    float tmp, iv, ft;
    unsigned int bi, ici, oci, yi, xi, fyi, fxi;
    for(ici = 0; ici < ic; ici++){
        torch::TensorAccessor<float, 3> octa = ta[ici];
        for(oci = 0; oci < oc; oci++){
            torch::TensorAccessor<float, 2> fta = octa[oci];
            torch::TensorAccessor<float, 2> fa = wa[oci][ici];
            for(fyi = 0; fyi < ks; fyi++){
                torch::TensorAccessor<float, 1> ftra = fta[fyi];
                torch::TensorAccessor<float, 1> fra = fa[fyi];
                for(fxi = 0; fxi < ks; fxi++){
                    float w = fra[fxi];
                    if(w == 0){
                        ftra[fxi] = 0;
                    }else{
                        ftra[fxi] = fabs(threshold / w);
                    }
                }
            }
        }
    }

    for(bi = 0; bi < bs; bi++){
        torch::TensorAccessor<float, 3> ocaa = oa[bi];
        torch::TensorAccessor<float, 3> sa = ia[bi];
        for(ici = 0; ici < ic; ici++){
            torch::TensorAccessor<float, 2> ca = sa[ici];
            torch::TensorAccessor<float, 3> octa = ta[ici];
            for(oci = 0; oci < oc; oci++){
                torch::TensorAccessor<float, 2> oca = ocaa[oci];
                torch::TensorAccessor<float, 2> fta = octa[oci];
                torch::TensorAccessor<float, 2> fa = wa[oci][ici];
                for(yi = 0; yi < oys; yi++){
                    torch::TensorAccessor<float, 1> ocra = oca[yi];
                    for(xi = 0; xi < oxs; xi++){
                        tmp = 0;

                        for(fyi = 0; fyi < ks; fyi++){
                            torch::TensorAccessor<float, 1> cra = ca[(yi * stride) + (fyi * dilation)];
                            torch::TensorAccessor<float, 1> ftra = fta[fyi];
                            torch::TensorAccessor<float, 1> fra = fa[fyi];
                            for(fxi = 0; fxi < ks; fxi++){
                                iv = cra[(xi * stride) + (fxi * dilation)];
                                ft = ftra[fxi];

                                if(iv > ft || iv < -ft){
                                    tmp += iv * fra[fxi];
                                }
                            }
                        }

                        ocra[xi] += tmp;
                    }
                }
            }
        }
    }

    return ot;
}

std::tuple<at::Tensor, unsigned int, unsigned int, unsigned int> debug_rtp_conv2d(torch::Tensor &it, torch::Tensor &wt, unsigned int stride, unsigned int padding, unsigned int dilation, float threshold){
    it = torch::nn::functional::pad(
        it,
        torch::nn::functional::PadFuncOptions({padding, padding, padding, padding})
            .mode(torch::kConstant)
            .value(0)
    );
    
    unsigned int bs = it.size(0);
    unsigned int ic = it.size(1);
    unsigned int ys = it.size(2);
    unsigned int xs = it.size(3);
    unsigned int oc = wt.size(0);
    unsigned int ks = wt.size(2); //TODO: allow non-square convolutions

    unsigned int oys = ((ys - (((ks - 1) * dilation) + 1)) / stride) + 1;
    unsigned int oxs = ((xs - (((ks - 1) * dilation) + 1)) / stride) + 1;

    torch::Tensor tt = torch::empty({
        ic,
        oc,
        ks,
        ks
    });

    torch::Tensor ot = torch::zeros({
        bs,
        oc,
        oys,
        oxs
    });

    torch::TensorAccessor<float, 4> ta = tt.accessor<float, 4>();
    torch::TensorAccessor<float, 4> ia = it.accessor<float, 4>();
    torch::TensorAccessor<float, 4> wa = wt.accessor<float, 4>();
    torch::TensorAccessor<float, 4> oa = ot.accessor<float, 4>();

    unsigned int pruned_in_training = 0;
    unsigned int pruned_in_inference = 0;
    unsigned int total = bs * ic * oc * oys * oxs * ks * ks;

    float tmp, iv, ft, fv;
    unsigned int bi, ici, oci, yi, xi, fyi, fxi;
    for(ici = 0; ici < ic; ici++){
        torch::TensorAccessor<float, 3> octa = ta[ici];
        for(oci = 0; oci < oc; oci++){
            torch::TensorAccessor<float, 2> fta = octa[oci];
            torch::TensorAccessor<float, 2> fa = wa[oci][ici];
            for(fyi = 0; fyi < ks; fyi++){
                torch::TensorAccessor<float, 1> ftra = fta[fyi];
                torch::TensorAccessor<float, 1> fra = fa[fyi];
                for(fxi = 0; fxi < ks; fxi++){
                    float w = fra[fxi];
                    if(w == 0){
                        ftra[fxi] = 0;
                    }else{
                        ftra[fxi] = fabs(threshold / w);
                    }
                }
            }
        }
    }

    for(bi = 0; bi < bs; bi++){
        torch::TensorAccessor<float, 3> ocaa = oa[bi];
        torch::TensorAccessor<float, 3> sa = ia[bi];
        for(ici = 0; ici < ic; ici++){
            torch::TensorAccessor<float, 2> ca = sa[ici];
            torch::TensorAccessor<float, 3> octa = ta[ici];
            for(oci = 0; oci < oc; oci++){
                torch::TensorAccessor<float, 2> oca = ocaa[oci];
                torch::TensorAccessor<float, 2> fta = octa[oci];
                torch::TensorAccessor<float, 2> fa = wa[oci][ici];
                for(yi = 0; yi < oys; yi++){
                    torch::TensorAccessor<float, 1> ocra = oca[yi];
                    for(xi = 0; xi < oxs; xi++){
                        tmp = 0;

                        for(fyi = 0; fyi < ks; fyi++){
                            torch::TensorAccessor<float, 1> cra = ca[(yi * stride) + (fyi * dilation)];
                            torch::TensorAccessor<float, 1> ftra = fta[fyi];
                            torch::TensorAccessor<float, 1> fra = fa[fyi];
                            for(fxi = 0; fxi < ks; fxi++){
                                iv = cra[(xi * stride) + (fxi * dilation)];
                                ft = ftra[fxi];
                                fv = fra[fxi];

                                if(fv == 0){
                                    pruned_in_training++;
                                }else if(iv > ft || iv < -ft){
                                    tmp += iv * fv;
                                }else{
                                    pruned_in_inference++;
                                }
                            }
                        }

                        ocra[xi] += tmp;
                    }
                }
            }
        }
    }

    return std::make_tuple(ot, pruned_in_training, pruned_in_inference, total);
}

/*
***********************************
CONVOLUTIONAL 2D LAYERS (END)
***********************************
*/



/*
***********************************
CONVOLUTIONAL 1D LAYERS
***********************************
*/

torch::Tensor conv1d(torch::Tensor &it, torch::Tensor &wt, unsigned int stride, unsigned int padding, unsigned int dilation){
    it = torch::nn::functional::pad(
        it,
        torch::nn::functional::PadFuncOptions({padding, padding})
            .mode(torch::kConstant)
            .value(0)
    );
    
    unsigned int bs = it.size(0);
    unsigned int ic = it.size(1);
    unsigned int ys = it.size(2);
    unsigned int oc = wt.size(0);
    unsigned int ks = wt.size(2); //TODO: allow non-square convolutions

    unsigned int oys = ((ys - (((ks - 1) * dilation) + 1)) / stride) + 1;

    torch::Tensor ot = torch::zeros({
        bs,
        oc,
        oys
    });

    torch::TensorAccessor<float, 3> ia = it.accessor<float, 3>();
    torch::TensorAccessor<float, 3> wa = wt.accessor<float, 3>();
    torch::TensorAccessor<float, 3> oa = ot.accessor<float, 3>();

    float tmp;
    unsigned int bi, ici, oci, yi, fyi;
    for(bi = 0; bi < bs; bi++){
        torch::TensorAccessor<float, 2> ocaa = oa[bi];
        torch::TensorAccessor<float, 2> sa = ia[bi];
        for(ici = 0; ici < ic; ici++){
            torch::TensorAccessor<float, 1> ca = sa[ici];
            for(oci = 0; oci < oc; oci++){
                torch::TensorAccessor<float, 1> oca = ocaa[oci];
                torch::TensorAccessor<float, 1> fa = wa[oci][ici];
                for(yi = 0; yi < oys; yi++){
                    tmp = 0;

                    for(fyi = 0; fyi < ks; fyi++){
                        tmp += ca[(yi * stride) + (fyi * dilation)] * fa[fyi];
                    }

                    oca[yi] += tmp;
                }
            }
        }
    }

    return ot;
}

torch::Tensor rtp_conv1d(torch::Tensor &it, torch::Tensor &wt, unsigned int stride, unsigned int padding, unsigned int dilation, float threshold){
    it = torch::nn::functional::pad(
        it,
        torch::nn::functional::PadFuncOptions({padding, padding})
            .mode(torch::kConstant)
            .value(0)
    );
    
    unsigned int bs = it.size(0);
    unsigned int ic = it.size(1);
    unsigned int ys = it.size(2);
    unsigned int oc = wt.size(0);
    unsigned int ks = wt.size(2); //TODO: allow non-square convolutions

    unsigned int oys = ((ys - (((ks - 1) * dilation) + 1)) / stride) + 1;

    torch::Tensor tt = torch::empty({
        ic,
        oc,
        ks
    });

    torch::Tensor ot = torch::zeros({
        bs,
        oc,
        oys
    });

    torch::TensorAccessor<float, 3> ta = tt.accessor<float, 3>();
    torch::TensorAccessor<float, 3> ia = it.accessor<float, 3>();
    torch::TensorAccessor<float, 3> wa = wt.accessor<float, 3>();
    torch::TensorAccessor<float, 3> oa = ot.accessor<float, 3>();

    float tmp, iv, ft;
    unsigned int bi, ici, oci, yi, fyi;
    for(ici = 0; ici < ic; ici++){
        torch::TensorAccessor<float, 2> octa = ta[ici];
        for(oci = 0; oci < oc; oci++){
            torch::TensorAccessor<float, 1> fta = octa[oci];
            torch::TensorAccessor<float, 1> fa = wa[oci][ici];
            for(fyi = 0; fyi < ks; fyi++){
                float w = fa[fyi];
                if(w == 0){
                    fta[fyi] = 0;
                }else{
                    fta[fyi] = fabs(threshold / w);
                }
            }
        }
    }

    for(bi = 0; bi < bs; bi++){
        torch::TensorAccessor<float, 2> ocaa = oa[bi];
        torch::TensorAccessor<float, 2> sa = ia[bi];
        for(ici = 0; ici < ic; ici++){
            torch::TensorAccessor<float, 1> ca = sa[ici];
            torch::TensorAccessor<float, 2> octa = ta[ici];
            for(oci = 0; oci < oc; oci++){
                torch::TensorAccessor<float, 1> oca = ocaa[oci];
                torch::TensorAccessor<float, 1> fta = octa[oci];
                torch::TensorAccessor<float, 1> fa = wa[oci][ici];
                for(yi = 0; yi < oys; yi++){
                    tmp = 0;

                    for(fyi = 0; fyi < ks; fyi++){
                        iv = ca[(yi * stride) + (fyi * dilation)];
                        ft = fta[fyi];

                        if(iv > ft || iv < -ft){
                            tmp += iv * fa[fyi];
                        }
                    }

                    oca[yi] += tmp;
                }
            }
        }
    }

    return ot;
}

std::tuple<at::Tensor, unsigned int, unsigned int, unsigned int> debug_rtp_conv1d(torch::Tensor &it, torch::Tensor &wt, unsigned int stride, unsigned int padding, unsigned int dilation, float threshold){
    it = torch::nn::functional::pad(
        it,
        torch::nn::functional::PadFuncOptions({padding, padding})
            .mode(torch::kConstant)
            .value(0)
    );
    
    unsigned int bs = it.size(0);
    unsigned int ic = it.size(1);
    unsigned int ys = it.size(2);
    unsigned int oc = wt.size(0);
    unsigned int ks = wt.size(2); //TODO: allow non-square convolutions

    unsigned int oys = ((ys - (((ks - 1) * dilation) + 1)) / stride) + 1;

    torch::Tensor tt = torch::empty({
        ic,
        oc,
        ks
    });

    torch::Tensor ot = torch::zeros({
        bs,
        oc,
        oys
    });

    torch::TensorAccessor<float, 3> ta = tt.accessor<float, 3>();
    torch::TensorAccessor<float, 3> ia = it.accessor<float, 3>();
    torch::TensorAccessor<float, 3> wa = wt.accessor<float, 3>();
    torch::TensorAccessor<float, 3> oa = ot.accessor<float, 3>();

    unsigned int pruned_in_training = 0;
    unsigned int pruned_in_inference = 0;
    unsigned int total = bs * ic * oc * oys * ks;

    float tmp, iv, ft, fv;
    unsigned int bi, ici, oci, yi, fyi;
    for(ici = 0; ici < ic; ici++){
        torch::TensorAccessor<float, 2> octa = ta[ici];
        for(oci = 0; oci < oc; oci++){
            torch::TensorAccessor<float, 1> fta = octa[oci];
            torch::TensorAccessor<float, 1> fa = wa[oci][ici];
            for(fyi = 0; fyi < ks; fyi++){
                float w = fa[fyi];
                if(w == 0){
                    fta[fyi] = 0;
                }else{
                    fta[fyi] = fabs(threshold / w);
                }
            }
        }
    }

    for(bi = 0; bi < bs; bi++){
        torch::TensorAccessor<float, 2> ocaa = oa[bi];
        torch::TensorAccessor<float, 2> sa = ia[bi];
        for(ici = 0; ici < ic; ici++){
            torch::TensorAccessor<float, 1> ca = sa[ici];
            torch::TensorAccessor<float, 2> octa = ta[ici];
            for(oci = 0; oci < oc; oci++){
                torch::TensorAccessor<float, 1> oca = ocaa[oci];
                torch::TensorAccessor<float, 1> fta = octa[oci];
                torch::TensorAccessor<float, 1> fa = wa[oci][ici];
                for(yi = 0; yi < oys; yi++){
                    tmp = 0;

                    for(fyi = 0; fyi < ks; fyi++){
                        iv = ca[(yi * stride) + (fyi * dilation)];
                        ft = fta[fyi];
                        fv = fa[fyi];

                        if(fv == 0){
                            pruned_in_training++;
                        }else if(iv > ft || iv < -ft){
                            tmp += iv * fv;
                        }else{
                            pruned_in_inference++;
                        }
                    }

                    oca[yi] += tmp;
                }
            }
        }
    }

    return std::make_tuple(ot, pruned_in_training, pruned_in_inference, total);
}

/*
***********************************
CONVOLUTIONAL 1D LAYERS (END)
***********************************
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("linear", &linear, "This is rtp's no-runtime-pruning linear layer forwards function.");
  m.def("rtp_linear", &rtp_linear, "This is rtp's linear layer forwards function.");
  m.def("debug_rtp_linear", &debug_rtp_linear, "This is the debug version of rtp's linear layer forwards function.");
  
  m.def("conv2d", &conv2d, "This is rtp's no-runtime-pruning convolution 2d layer forwards function.");
  m.def("rtp_conv2d", &rtp_conv2d, "This is rtp's convolution 2d layer forwards function.");
  m.def("debug_rtp_conv2d", &debug_rtp_conv2d, "This is debug version of rtp's convolution 2d layer forwards function.");
  
  m.def("conv1d", &conv1d, "This is rtp's no-runtime-pruning convolution 1d layer forwards function.");
  m.def("rtp_conv1d", &rtp_conv1d, "This is rtp's convolution 1d layer forwards function.");
  m.def("debug_rtp_conv1d", &debug_rtp_conv1d, "This is debug version of rtp's convolution 1d layer forwards function.");
}