/* 
 * Copyright (C) 2017 Universitat Politècnica de València
 *
 * This file is part of FMAP.
 *
 * FMAP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * FMAP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with FMAP. If not, see <http://www.gnu.org/licenses/>.
 */
package org.agreement_technologies.service.map_dtg;

import org.agreement_technologies.common.map_dtg.DTGFactory;
import org.agreement_technologies.common.map_dtg.DTGSet;
import org.agreement_technologies.common.map_grounding.GroundedTask;

/**
 * DTGFactoryImp class creates the set of Domain Transition Graphs (one per
 * variable) from the grounded task.
 *
 * @author Oscar Sapena
 * @author Alejandro Torreno
 * @version %I%, %G%
 * @since 1.0
 */
public class DTGFactoryImp implements DTGFactory {

    /**
     * Creates the set of Domain Transition Graphs (one per variable) from the
     * grounded task.
     *
     * @param task Grounded task
     * @return Set of DTGs
     * @since 1.0
     */
    @Override
    public DTGSet create(GroundedTask task) {
        return new DTGSetImp(task);
    }

}
